"""
VoiceGuard — Speaker Identification Engine v3
==============================================
Uses Resemblyzer's pre-trained GE2E (d-vector) model — the same class of
model used in Google's speaker-diarization pipeline.

Pipeline:
  1. Load & resample audio to 16 kHz mono
  2. VOice Activity Detection (webrtcvad) — strip silence frames
  3. Split into overlapping 1.6-second segments
  4. Run each segment through Resemblyzer encoder (256-dim d-vector per segment)
  5. Enrollment: store the mean d-vector per speaker
  6. Identification: cosine similarity vs every enrolled speaker + confidence mapping

Why this works where Shazam-style hashing fails
------------------------------------------------
Resemblyzer is trained with a generalized end-to-end (GE2E) loss that explicitly
optimizes within-speaker compactness and between-speaker separation.
The resulting 256-dim d-vectors cluster tightly for the same speaker
regardless of what words are spoken, and stay far apart for different speakers.
Cosine similarity between same-speaker d-vectors is typically 0.85–0.99;
different-speaker pairs sit at 0.40–0.75. The gap is large enough for reliable
text-independent identification with only a single enrollment sample.

Features still surfaced to the UI (for voiceprint analysis panel):
  n_peaks, n_fingerprints  — MFCC peak characteristics for display
  highest/lowest freq peak — spectral extremes
  spectral_similarity      — cosine similarity 
  t_score                  — separation margin between #1 and #2 match
"""

import os
import json
import pickle
import struct
import collections
import numpy as np
import librosa
import librosa.feature
from pathlib import Path

# ─── Optional webrtcvad (soft dependency) ─────────────────────────────────────
try:
    import webrtcvad
    _HAVE_VAD = True
except ImportError:
    _HAVE_VAD = False

# ─── Resemblyzer ──────────────────────────────────────────────────────────────
from resemblyzer import VoiceEncoder, preprocess_wav

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
SPEAKERS_FILE     = os.path.join(BASE_DIR, "data", "speakers.json")
EMBEDDINGS_FILE   = os.path.join(BASE_DIR, "embeddings", "speaker_embeddings.pkl")
VOICE_SAMPLES_DIR = os.path.join(BASE_DIR, "voice_samples")

# ─── Audio Parameters ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
FMIN        = 80
FMAX        = 7600
N_MFCC      = 40

# ─── Matching thresholds ──────────────────────────────────────────────────────
# Resemblyzer cosine similarities (pre-recorded WAV vs live WebRTC mic):
#   same speaker (clean WAV)   → ~0.80–0.99
#   same speaker (live mic)    → ~0.60–0.85  ← channel mismatch lowers scores
#   diff speaker               → ~0.35–0.65
# We accept a match when:
#   (a) raw cosine ≥ ABS_THRESHOLD   AND
#   (b) margin over 2nd-best ≥ MARGIN_THRESHOLD
ABS_THRESHOLD    = 0.60   # absolute cosine floor (lowered for WebRTC channel mismatch)
MARGIN_THRESHOLD = 0.06   # best must beat 2nd-best by at least this much

# ─── Lazy-load the encoder (one instance for the whole process) ───────────────
_encoder: VoiceEncoder | None = None

def get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


# ══════════════════════════════════════════════════════════════════════════════
#  I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_speakers():
    """Load speaker metadata from speakers.json."""
    with open(SPEAKERS_FILE, "r") as f:
        return json.load(f)


def load_embeddings():
    """Load d-vector database. Returns None if not yet enrolled."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return None
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  VAD-based silence removal
# ══════════════════════════════════════════════════════════════════════════════

def _vad_strip(y: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:
    """
    Remove non-speech frames using WebRTC VAD.
    Falls back to returning the original signal if webrtcvad isn't available
    or the audio is too short.
    """
    if not _HAVE_VAD or len(y) < sr * 0.3:
        return y

    try:
        vad = webrtcvad.Vad(aggressiveness)
        frame_ms  = 30          # 10 | 20 | 30 ms frames supported by webrtcvad
        frame_len = int(sr * frame_ms / 1000)

        # webrtcvad needs 16-bit PCM bytes
        pcm = (y * 32767).astype(np.int16)
        speech_frames = []
        for i in range(0, len(pcm) - frame_len, frame_len):
            frame = pcm[i : i + frame_len]
            raw   = struct.pack(f"{len(frame)}h", *frame)
            if vad.is_speech(raw, sr):
                speech_frames.append(y[i : i + frame_len])

        if len(speech_frames) < 3:
            return y  # not enough speech detected — return original
        return np.concatenate(speech_frames)
    except Exception:
        return y


# ══════════════════════════════════════════════════════════════════════════════
#  Compute display statistics (for the Voiceprint Analysis panel)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_display_stats(y: np.ndarray, sr: int) -> dict:
    """
    Compute MFCC-based statistics purely for display in the UI panel.
    These are NOT used for matching.
    """
    from scipy.ndimage import maximum_filter

    # Log-mel spectrogram
    stft    = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann')
    mag     = np.abs(stft)
    mel_fb  = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    log_mel = librosa.amplitude_to_db(np.dot(mel_fb, mag), ref=np.max)

    # Peak detection
    nbhd_max = maximum_filter(log_mel, size=15, mode='constant', cval=log_mel.min() - 1)
    is_peak  = (log_mel == nbhd_max) & (log_mel > -60)
    freq_idx, time_idx = np.where(is_peak)
    amps = log_mel[freq_idx, time_idx]
    order = np.argsort(amps)[::-1][:500]
    freq_idx, time_idx, amps = freq_idx[order], time_idx[order], amps[order]

    peaks = sorted(zip(time_idx.tolist(), freq_idx.tolist(), amps.tolist()), key=lambda p: p[0])
    highest = max(peaks, key=lambda p: p[1]) if peaks else None
    lowest  = min(peaks, key=lambda p: p[1]) if peaks else None

    # Pentagon hash count (display only)
    fp = {}
    for i, (t_a, f_a, _) in enumerate(peaks):
        found = 0
        for j in range(i + 1, len(peaks)):
            t_t, f_t, _ = peaks[j]
            dt = t_t - t_a
            if dt > 100:
                break
            fp.setdefault(((f_a // 2) * 10_000_000 + (f_t // 2) * 100_000 + dt // 2), []).append(t_a)
            found += 1
            if found >= 5:
                break

    def mel_hz(f):
        lo, hi = librosa.hz_to_mel(FMIN), librosa.hz_to_mel(FMAX)
        return float(librosa.mel_to_hz(lo + f * (hi - lo) / max(N_MELS - 1, 1)))

    def fmt(peak):
        if peak is None:
            return None
        t, f, amp = peak
        return {"time_frame": t, "freq_bin": f, "freq_hz": round(mel_hz(f), 1), "amp_db": round(float(amp), 1)}

    return {
        "n_peaks":        len(peaks),
        "n_fingerprints": len(fp),
        "highest_peak":   fmt(highest),
        "lowest_peak":    fmt(lowest),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  D-Vector Embedding (Resemblyzer)
# ══════════════════════════════════════════════════════════════════════════════

def _enhance_audio(wav: np.ndarray) -> np.ndarray:
    """
    Apply pre-emphasis + per-utterance CMVN-style normalization to reduce
    channel/microphone mismatch between enrollment WAV and live WebRTC audio.

    Pre-emphasis boosts high-frequency formants that matter for speaker ID
    and partially compensates for the lossy Opus codec used in WebRTC.
    """
    # Pre-emphasis filter: y[n] = x[n] - 0.97 * x[n-1]
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    # Peak-normalize
    peak = np.max(np.abs(wav))
    if peak > 1e-6:
        wav = wav / peak
    return wav.astype(np.float32)


def embed_audio(audio_path: str) -> np.ndarray:
    """
    Load a WAV file, apply VAD + audio enhancement, and compute a single
    256-dim d-vector by averaging embeddings across overlapping 1.6-second windows.

    Returns
    -------
    np.ndarray  shape (256,), L2-normalized
    """
    enc = get_encoder()

    # Resemblyzer's preprocess_wav handles resampling + peak-normalization
    wav = preprocess_wav(audio_path)

    # Apply pre-emphasis to compensate for WebRTC/Opus codec channel mismatch
    wav = _enhance_audio(wav)

    # Extra VAD pass to strip silence (improves enrollment quality)
    wav = _vad_strip(wav, SAMPLE_RATE)

    if len(wav) < SAMPLE_RATE * 0.5:
        raise ValueError("Audio too short after silence removal — please speak for at least 2 seconds.")

    # embed_utterance returns the mean d-vector over all sliding windows
    dvec = enc.embed_utterance(wav)       # shape (256,)
    dvec = dvec / (np.linalg.norm(dvec) + 1e-9)
    return dvec


# ══════════════════════════════════════════════════════════════════════════════
#  Enrollment
# ══════════════════════════════════════════════════════════════════════════════

def enroll_speakers():
    """
    Generate 256-dim d-vector embeddings for all registered speakers.
    Saves to embeddings/speaker_embeddings.pkl.
    Returns number of successfully enrolled speakers.
    """
    speakers     = load_speakers()
    embeddings   = {}
    enrolled_cnt = 0

    for speaker in speakers:
        sample_path = os.path.join(BASE_DIR, speaker["voice_sample"])
        if not os.path.exists(sample_path):
            print(f"⚠️  Voice sample not found: {sample_path}")
            continue

        try:
            print(f"🔍 Enrolling: {speaker['name']}...")
            dvec  = embed_audio(sample_path)
            stats = _compute_display_stats(
                librosa.load(sample_path, sr=SAMPLE_RATE, mono=True)[0],
                SAMPLE_RATE
            )

            embeddings[speaker["id"]] = {
                "name":           speaker["name"],
                "roll_number":    speaker["roll_number"],
                "branch":         speaker["branch"],
                "dvector":        dvec,
                "n_peaks":        stats["n_peaks"],
                "n_fingerprints": stats["n_fingerprints"],
                "highest_peak":   stats["highest_peak"],
                "lowest_peak":    stats["lowest_peak"],
            }
            enrolled_cnt += 1
            print(f"   ✅ {speaker['name']} — d-vec norm={np.linalg.norm(dvec):.4f}, "
                  f"peaks={stats['n_peaks']}, fp={stats['n_fingerprints']}")

        except Exception as e:
            import traceback
            print(f"   ❌ Error enrolling {speaker['name']}: {e}")
            traceback.print_exc()

    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\n📦 Saved d-vectors for {enrolled_cnt} speakers → {EMBEDDINGS_FILE}")
    return enrolled_cnt


# ══════════════════════════════════════════════════════════════════════════════
#  Identification
# ══════════════════════════════════════════════════════════════════════════════

def identify_speaker(audio_path: str) -> dict:
    """
    Identify the speaker using deep d-vector cosine similarity.

    Returns dict:
        matched      (bool)   — True if best score passes both thresholds
        confidence   (float)  — Display confidence 0–100
        threshold    (float)  — ABS_THRESHOLD × 100 for UI display
        speaker      (dict)   — Best-match speaker info or None
        all_scores   (list)   — All speakers with individual scores
        analysis     (dict)   — Peak/fingerprint stats + frequency extremes
    """
    embeddings = load_embeddings()

    if embeddings is None or len(embeddings) == 0:
        return {
            "matched": False, "confidence": 0,
            "threshold": ABS_THRESHOLD * 100,
            "speaker": None, "all_scores": [], "analysis": {},
            "error": "No enrolled speakers found. Please enroll speakers first.",
        }

    # ── Embed the incoming audio ─────────────────────────────────────────────
    try:
        query_dvec = embed_audio(audio_path)
    except Exception as e:
        return {
            "matched": False, "confidence": 0,
            "threshold": ABS_THRESHOLD * 100,
            "speaker": None, "all_scores": [], "analysis": {},
            "error": f"Audio processing error: {e}",
        }

    # ── Display stats for the voiceprint-analysis panel ─────────────────────
    try:
        y_raw, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        display  = _compute_display_stats(y_raw, SAMPLE_RATE)
    except Exception:
        display  = {"n_peaks": 0, "n_fingerprints": 0, "highest_peak": None, "lowest_peak": None}

    if display["n_peaks"] < 5:
        return {
            "matched": False, "confidence": 0,
            "threshold": ABS_THRESHOLD * 100,
            "speaker": None, "all_scores": [], "analysis": display,
            "error": "Audio too short or too quiet — please speak for at least 3 seconds.",
        }

    # ── Cosine similarity against every enrolled speaker ─────────────────────
    raw_scores = []
    for sid, enrolled in embeddings.items():
        sim = float(np.dot(query_dvec, enrolled["dvector"]))
        raw_scores.append({
            "id":          sid,
            "name":        enrolled["name"],
            "roll_number": enrolled["roll_number"],
            "branch":      enrolled["branch"],
            "raw_sim":     sim,
        })

    raw_scores.sort(key=lambda x: x["raw_sim"], reverse=True)
    best   = raw_scores[0]
    second = raw_scores[1]["raw_sim"] if len(raw_scores) > 1 else 0.0
    margin = best["raw_sim"] - second

    # ── Convert raw cosine similarity to display confidence ──────────────────
    #   Resemblyzer same-speaker: ~0.72–0.99  →  we map [ABS_THRESHOLD, 1.0] → [60, 100]
    #   Different speakers:       ~0.40–0.72  →  maps to <60
    def sim_to_conf(sim):
        # Map [ABS_THRESHOLD - 0.15, 1.0] → [0, 100]
        # Genuine speaker (live mic): ~0.60–0.85 → shows 60–90%
        # Impostor:                   ~0.35–0.59 → shows <50%
        lo, hi = ABS_THRESHOLD - 0.15, 1.0
        pct = (sim - lo) / max(hi - lo, 1e-9)
        return round(max(0.0, min(100.0, pct * 100.0)), 2)

    all_scores = []
    for s in raw_scores:
        all_scores.append({
            "id":          s["id"],
            "name":        s["name"],
            "roll_number": s["roll_number"],
            "branch":      s["branch"],
            "confidence":  sim_to_conf(s["raw_sim"]),
            "raw_sim":     round(s["raw_sim"] * 100, 2),
        })

    best_conf = all_scores[0]["confidence"]

    # ── Match decision ────────────────────────────────────────────────────────
    #   Both conditions must pass:
    #   (a) absolute cosine ≥ ABS_THRESHOLD  (rules out low-quality audio / strangers)
    #   (b) margin ≥ MARGIN_THRESHOLD        (rules out ambiguous ties between speakers)
    is_matched = (best["raw_sim"] >= ABS_THRESHOLD) and (margin >= MARGIN_THRESHOLD)

    # T-score equivalent for display (sigma above cohort mean)
    others_sims = [s["raw_sim"] for s in raw_scores[1:]]
    mu  = float(np.mean(others_sims)) if others_sims else 0.0
    sig = float(np.std(others_sims))  if others_sims else 1e-9
    sig = max(sig, 1e-9)
    t_score = (best["raw_sim"] - mu) / sig

    return {
        "matched":    is_matched,
        "confidence": best_conf,
        "threshold":  ABS_THRESHOLD * 100,
        "speaker":    all_scores[0] if is_matched else None,
        "all_scores": all_scores,
        "analysis": {
            "n_peaks":        display["n_peaks"],
            "n_fingerprints": display["n_fingerprints"],
            "t_score":        round(t_score, 3),
            "highest_peak":   display["highest_peak"],
            "lowest_peak":    display["lowest_peak"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — run directly to enroll speakers
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🎤 VoiceGuard — D-Vector Speaker Embedding Engine v3")
    print("=" * 55)
    print("\nEnrolling speakers from voice samples...\n")
    count = enroll_speakers()
    print(f"\n✅ Done! {count} speakers enrolled.")
