/**
 * VoiceGuard — Frontend Application
 * Handles mic capture, API communication, and UI state management.
 */

// ---- DOM Elements ----
const micButton = document.getElementById("mic-button");
const micWrapper = document.getElementById("mic-wrapper");
const authInstruction = document.getElementById("auth-instruction");
const timerContainer = document.getElementById("timer-container");
const timerFill = document.getElementById("timer-fill");
const timerText = document.getElementById("timer-text");
const waveformContainer = document.getElementById("waveform-container");
const waveformCanvas = document.getElementById("waveform-canvas");
const processing = document.getElementById("processing");
const authContainer = document.querySelector(".auth-container");
const resultContainer = document.getElementById("result-container");
const resultGranted = document.getElementById("result-granted");
const resultDenied = document.getElementById("result-denied");
const resultError = document.getElementById("result-error");
const btnRetry = document.getElementById("btn-retry");
const btnEnroll = document.getElementById("btn-enroll");
const btnClearLog = document.getElementById("btn-clear-log");
const speakersList = document.getElementById("speakers-list");
const logsList = document.getElementById("logs-list");
const systemStatus = document.getElementById("system-status");

// ---- State ----
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer = null;
let timerInterval = null;
let analyserNode = null;
let animationFrameId = null;
let audioContext = null;

const RECORDING_DURATION = 5; // seconds

// ---- Initialize ----
document.addEventListener("DOMContentLoaded", () => {
    loadSpeakers();
    setupEventListeners();
});

function setupEventListeners() {
    micButton.addEventListener("click", toggleRecording);
    btnRetry.addEventListener("click", resetToReady);
    btnEnroll.addEventListener("click", enrollSpeakers);
    btnClearLog.addEventListener("click", clearLogs);
}

// ---- Load Speakers ----
async function loadSpeakers() {
    try {
        const response = await fetch("/speakers");
        const data = await response.json();

        if (data.success && data.speakers.length > 0) {
            renderSpeakers(data.speakers);
            const enrolledCount = data.speakers.filter(s => s.enrolled).length;
            systemStatus.textContent = `${enrolledCount}/${data.speakers.length} Enrolled`;
        } else {
            speakersList.innerHTML = `
                <div class="empty-logs">
                    <p>No speakers registered yet</p>
                </div>
            `;
        }
    } catch (err) {
        console.error("Failed to load speakers:", err);
        speakersList.innerHTML = `
            <div class="empty-logs">
                <p>Failed to load speakers</p>
            </div>
        `;
    }
}

function renderSpeakers(speakers) {
    speakersList.innerHTML = speakers.map(speaker => {
        const initials = speaker.name
            .split(" ")
            .map(w => w[0])
            .join("")
            .slice(0, 2);

        const statusClass = speaker.enrolled ? "enrolled" : "not-enrolled";

        return `
            <div class="speaker-card" data-id="${speaker.id}">
                <div class="speaker-avatar">${initials}</div>
                <div class="speaker-info">
                    <div class="speaker-name">${speaker.name}</div>
                    <div class="speaker-meta">${speaker.roll_number} • ${speaker.branch}</div>
                </div>
                <div class="speaker-status ${statusClass}" title="${speaker.enrolled ? 'Enrolled' : 'Not enrolled'}"></div>
            </div>
        `;
    }).join("");
}

// ---- Enrollment ----
async function enrollSpeakers() {
    btnEnroll.classList.add("enrolling");
    systemStatus.textContent = "Enrolling...";

    try {
        const response = await fetch("/enroll", { method: "POST" });
        const data = await response.json();

        if (data.success) {
            systemStatus.textContent = `${data.count} Enrolled ✓`;
            loadSpeakers(); // Refresh the list
        } else {
            systemStatus.textContent = "Enroll Failed";
            console.error("Enrollment failed:", data.error);
        }
    } catch (err) {
        systemStatus.textContent = "Enroll Error";
        console.error("Enrollment error:", err);
    } finally {
        btnEnroll.classList.remove("enrolling");
    }
}

// ---- Recording ----
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        // Set up audio context for waveform visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);

        // Set up MediaRecorder
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: getSupportedMimeType()
        });

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            processRecording();
        };

        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;

        // Update UI
        micWrapper.classList.add("recording");
        authInstruction.textContent = "Listening... Speak clearly into the microphone";
        timerContainer.classList.add("visible");
        waveformContainer.classList.add("visible");

        // Start waveform visualization
        drawWaveform();

        // Start countdown timer
        let elapsed = 0;
        timerFill.style.width = "0%";
        timerText.textContent = `${RECORDING_DURATION}s`;

        timerInterval = setInterval(() => {
            elapsed += 0.1;
            const progress = (elapsed / RECORDING_DURATION) * 100;
            timerFill.style.width = `${Math.min(progress, 100)}%`;
            timerText.textContent = `${Math.max(0, (RECORDING_DURATION - elapsed)).toFixed(1)}s`;
        }, 100);

        // Auto-stop after duration
        recordingTimer = setTimeout(() => {
            stopRecording();
        }, RECORDING_DURATION * 1000);

    } catch (err) {
        console.error("Microphone access denied:", err);
        showError("Microphone access denied. Please allow microphone access and try again.");
    }
}

function stopRecording() {
    if (!isRecording) return;

    isRecording = false;
    clearTimeout(recordingTimer);
    clearInterval(timerInterval);
    cancelAnimationFrame(animationFrameId);

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    // Update UI
    micWrapper.classList.remove("recording");
    timerContainer.classList.remove("visible");
    waveformContainer.classList.remove("visible");
    timerFill.style.width = "100%";

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
}

function getSupportedMimeType() {
    const types = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
    ];
    for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return "audio/webm";
}

// ---- Process Recording ----
async function processRecording() {
    if (audioChunks.length === 0) {
        showError("No audio recorded. Please try again.");
        return;
    }

    // Show processing state
    authContainer.classList.add("hidden");
    processing.classList.add("visible");
    resultContainer.classList.remove("visible");

    try {
        const audioBlob = new Blob(audioChunks, { type: getSupportedMimeType() });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        const response = await fetch("/authenticate", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        processing.classList.remove("visible");

        if (data.success) {
            showResult(data);
        } else {
            showError(data.error || "Authentication failed. Please try again.");
        }
    } catch (err) {
        processing.classList.remove("visible");
        showError("Network error. Please check if the server is running.");
        console.error("Authentication error:", err);
    }
}

// ---- Show Results ----
function showResult(data) {
    resultContainer.classList.add("visible");

    // Hide all result cards first
    resultGranted.classList.remove("visible");
    resultDenied.classList.remove("visible");
    resultError.classList.remove("visible");

    if (data.matched && data.speaker) {
        // Access Granted
        resultGranted.classList.add("visible");

        document.getElementById("detail-name").textContent = data.speaker.name;
        document.getElementById("detail-roll").textContent = data.speaker.roll_number;
        document.getElementById("detail-branch").textContent = data.speaker.branch;
        document.getElementById("confidence-granted").textContent = `${data.confidence}%`;

        // Animate confidence bar
        setTimeout(() => {
            document.getElementById("confidence-fill-granted").style.width = `${data.confidence}%`;
        }, 300);

        // Render fingerprint analysis badge
        renderAnalysisBadge("analysis-badge-granted", data);

        // Add to access log
        addLogEntry("granted", data.speaker.name, data.confidence);

        // Highlight matched speaker card
        highlightSpeaker(data.speaker.id);

    } else {
        // Access Denied
        resultDenied.classList.add("visible");

        document.getElementById("confidence-denied").textContent = `${data.confidence}%`;
        document.getElementById("threshold-info").textContent = `Threshold: ${data.threshold}%`;

        // Animate confidence bar
        setTimeout(() => {
            document.getElementById("confidence-fill-denied").style.width = `${data.confidence}%`;
        }, 300);

        // Render fingerprint analysis badge
        renderAnalysisBadge("analysis-badge-denied", data);

        // Add to access log
        addLogEntry("denied", "Unknown Speaker", data.confidence);
    }
}

// ---- Fingerprint Analysis Badge ----
function renderAnalysisBadge(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const a       = data.analysis || {};
    const speaker = data.all_scores && data.all_scores[0];

    const hiPeak  = a.highest_peak;
    const loPeak  = a.lowest_peak;
    const hiHz    = hiPeak ? `${hiPeak.freq_hz} Hz` : "—";
    const loHz    = loPeak ? `${loPeak.freq_hz} Hz` : "—";

    const rawSim  = speaker ? `${speaker.raw_sim}%` : "—";
    const tScore  = (a.t_score !== undefined) ? `${a.t_score.toFixed(3)}σ` : "—";

    container.innerHTML = `
        <div class="analysis-badge">
            <div class="analysis-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                Voiceprint Analysis
            </div>
            <div class="analysis-grid">
                <div class="analysis-stat">
                    <span class="astat-label">Constellation Peaks</span>
                    <span class="astat-value">${a.n_peaks ?? "—"}</span>
                </div>
                <div class="analysis-stat">
                    <span class="astat-label">Pentagon Hashes</span>
                    <span class="astat-value">${a.n_fingerprints ?? "—"}</span>
                </div>
                <div class="analysis-stat">
                    <span class="astat-label">Spectral Similarity</span>
                    <span class="astat-value accent-cyan">${rawSim}</span>
                </div>
                <div class="analysis-stat">
                    <span class="astat-label">T-Score (Separation)</span>
                    <span class="astat-value accent-cyan">${tScore}</span>
                </div>
                <div class="analysis-stat">
                    <span class="astat-label">Highest Freq Peak</span>
                    <span class="astat-value">${hiHz}</span>
                </div>
                <div class="analysis-stat">
                    <span class="astat-label">Lowest Freq Peak</span>
                    <span class="astat-value">${loHz}</span>
                </div>
            </div>
        </div>
    `;
}

function showError(message) {
    processing.classList.remove("visible");
    authContainer.classList.add("hidden");
    resultContainer.classList.add("visible");

    resultGranted.classList.remove("visible");
    resultDenied.classList.remove("visible");
    resultError.classList.add("visible");

    document.getElementById("error-message").textContent = message;

    addLogEntry("error", message, null);
}

function highlightSpeaker(speakerId) {
    // Remove previous highlights
    document.querySelectorAll(".speaker-card").forEach(card => {
        card.style.borderColor = "";
        card.style.background = "";
    });

    const matched = document.querySelector(`.speaker-card[data-id="${speakerId}"]`);
    if (matched) {
        matched.style.borderColor = "rgba(0, 230, 118, 0.4)";
        matched.style.background = "rgba(0, 230, 118, 0.06)";
        matched.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

// ---- Reset ----
function resetToReady() {
    // Reset all result UI
    resultContainer.classList.remove("visible");
    resultGranted.classList.remove("visible");
    resultDenied.classList.remove("visible");
    resultError.classList.remove("visible");
    processing.classList.remove("visible");
    authContainer.classList.remove("hidden");

    // Reset confidence bars
    document.getElementById("confidence-fill-granted").style.width = "0%";
    document.getElementById("confidence-fill-denied").style.width = "0%";

    // Reset instruction text
    authInstruction.textContent = "Press the microphone to begin voice authentication";

    // Remove speaker highlights
    document.querySelectorAll(".speaker-card").forEach(card => {
        card.style.borderColor = "";
        card.style.background = "";
    });
}

// ---- Access Log ----
function addLogEntry(type, title, confidence) {
    // Remove empty state
    const emptyState = logsList.querySelector(".empty-logs");
    if (emptyState) emptyState.remove();

    const now = new Date();
    const timeStr = now.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });

    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.innerHTML = `
        <div class="log-dot ${type}"></div>
        <div class="log-info">
            <div class="log-title">${type === "granted" ? "✓ " : type === "denied" ? "✗ " : "⚠ "}${title}</div>
            <div class="log-time">${timeStr}</div>
        </div>
        ${confidence !== null ? `<div class="log-confidence">${confidence}%</div>` : ""}
    `;

    // Insert at the top
    logsList.insertBefore(entry, logsList.firstChild);

    // Keep max 50 entries
    while (logsList.children.length > 50) {
        logsList.removeChild(logsList.lastChild);
    }
}

function clearLogs() {
    logsList.innerHTML = `
        <div class="empty-logs">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="empty-icon">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 6v6l4 2"/>
            </svg>
            <p>No access attempts yet</p>
        </div>
    `;
}

// ---- Waveform Visualization ----
function drawWaveform() {
    if (!analyserNode) return;

    const ctx = waveformCanvas.getContext("2d");
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;

    function draw() {
        animationFrameId = requestAnimationFrame(draw);
        analyserNode.getByteTimeDomainData(dataArray);

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw background
        ctx.fillStyle = "rgba(10, 14, 26, 0.3)";
        ctx.fillRect(0, 0, width, height);

        // Draw waveform
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#00d4ff";
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(0, 212, 255, 0.5)";
        ctx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Draw center line
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
    }

    draw();
}
