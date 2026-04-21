"""
VoiceGuard — Speaker Authentication Web Server
Flask backend for the Shazam-style audio fingerprinting system.
Routes: / | /speakers | /authenticate | /enroll | /health
"""

import os
import uuid
import json
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
from speaker_engine import enroll_speakers, identify_speaker, load_speakers, load_embeddings

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/speakers", methods=["GET"])
def get_speakers():
    """Return the list of registered speakers."""
    try:
        speakers = load_speakers()
        # Check which speakers have been enrolled (have embeddings)
        embeddings = load_embeddings()
        enrolled_ids = set(embeddings.keys()) if embeddings else set()

        for speaker in speakers:
            speaker["enrolled"] = speaker["id"] in enrolled_ids

        return jsonify({"success": True, "speakers": speakers})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/authenticate", methods=["POST"])
def authenticate():
    """
    Receive audio from the browser, process it, and identify the speaker.
    """
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file received"}), 400

    audio_file = request.files["audio"]

    # Generate unique filenames for temp files
    file_id = str(uuid.uuid4())
    webm_path = os.path.join(TEMP_DIR, f"{file_id}.webm")
    wav_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # Save the uploaded webm file
        audio_file.save(webm_path)

        # Convert webm to wav using pydub (requires ffmpeg)
        audio = AudioSegment.from_file(webm_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # 16 kHz mono required by fingerprinting engine
        audio.export(wav_path, format="wav")

        # Run speaker identification
        result = identify_speaker(wav_path)

        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        # Clean up temp files
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.route("/enroll", methods=["POST"])
def enroll():
    """Re-generate embeddings from voice samples."""
    try:
        count = enroll_speakers()
        return jsonify({
            "success": True,
            "message": f"Successfully enrolled {count} speakers",
            "count": count,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — returns enrolled speaker count and fingerprint stats."""
    embeddings = load_embeddings()
    stats = []
    if embeddings:
        for sid, data in embeddings.items():
            stats.append({
                "id":          sid,
                "name":        data["name"],
                "n_peaks":     data.get("n_peaks", 0),
                "n_fingerprints": data.get("n_fingerprints", 0),
            })
    return jsonify({
        "status": "ok",
        "enrolled_speakers": len(embeddings) if embeddings else 0,
        "speaker_stats": stats,
    })


if __name__ == "__main__":
    # Auto-enroll on first run if embeddings don't exist
    if not os.path.exists(os.path.join(BASE_DIR, "embeddings", "speaker_embeddings.pkl")):
        print("\n🎤 First run detected! Enrolling speakers...")
        enroll_speakers()
        print()

    print("🚀 Starting VoiceGuard Server...")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5050)
