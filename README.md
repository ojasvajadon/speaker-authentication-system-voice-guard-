# speaker-authentication-system-voice-guard-
VoiceGuard — Speaker Authentication System  VoiceGuard is a lightweight speaker authentication system that identifies a person using their voice. Instead of relying on passwords or expensive biometric hardware, this project uses audio signal processing techniques to perform fast and reliable voice-based access control.  
The system follows a structured signal processing pipeline:

1. Audio Acquisition & Digitization

Voice input is captured via the browser and converted into a digital signal using a sampling rate of 16 kHz, ensuring compliance with the Nyquist criterion for speech signals.

2. Pre-processing
Noise reduction to suppress environmental disturbances
Amplitude normalization for consistent signal levels
Pre-emphasis filtering to enhance high-frequency components
3. Framing & Windowing

The speech signal is divided into short overlapping frames (~20–30 ms) to assume stationarity.
A Hamming window is applied to reduce spectral leakage and smooth frame boundaries.

4. Time-Frequency Analysis (STFT)

Short-Time Fourier Transform converts each frame into the frequency domain, producing a spectrogram that captures how frequencies evolve over time.

5. Feature Extraction (MFCC + Spectral Features)
MFCCs model human auditory perception
Spectral features capture frequency distribution
These features represent speaker-specific characteristics
6. Audio Fingerprinting (Key Innovation)

Instead of storing full features, the system extracts high-energy spectral peaks from the spectrogram and constructs a constellation map.

These peaks are converted into hashes using:

(f₁, f₂, Δt)

This creates a compact and robust fingerprint for each speaker.

7. Hash-Based Matching

During authentication:

Incoming audio is processed in the same way
Generated hashes are compared with stored hashes
Matching is performed using efficient O(1) lookup
8. Decision Logic
Spectral Similarity Score determines match strength
T-score measures separation from other speakers
A threshold (e.g., 15%) decides:
✅ Access Granted
❌ Access Denied
