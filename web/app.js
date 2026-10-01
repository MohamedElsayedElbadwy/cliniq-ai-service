const startCallButton = document.querySelector("#start-call-button");
const muteButton = document.querySelector("#mute-button");
const endCallButton = document.querySelector("#end-call-button");
const connectionStatus = document.querySelector("#connection-status");
const microphoneStatus = document.querySelector("#microphone-status");
const assistantStatus = document.querySelector("#assistant-status");
const transcript = document.querySelector("#transcript");
const errorMessage = document.querySelector("#error-message");

// Tuning values for automatic voice activity and silence detection.
const SPEECH_DETECTION_THRESHOLD = 0.018;
const SILENCE_DURATION_MS = 1000;
const MIN_RECORDING_DURATION_MS = 450;
const MIN_RECORDING_BYTES = 1000;
const ANALYSER_FFT_SIZE = 2048;

let microphoneStream;
let audioContext;
let analyser;
let recorder;
let recordedChunks = [];
let history = [];
let animationFrame;
let currentAudio;
let callActive = false;
let muted = false;
let processing = false;
let assistantSpeaking = false;
let speechStartedAt = 0;
let silenceStartedAt;

// Updates one visible call-status field without duplicating DOM logic.
function setStatus(element, value) {
  element.textContent = value;
}

// Displays the most recent recoverable Voice Call error to the user.
function reportError(message) {
  errorMessage.textContent = message;
}

// Adds one completed user or assistant message to the in-memory call transcript.
function addTranscript(role, text) {
  const empty = transcript.querySelector(".empty");
  if (empty) empty.remove();
  const item = document.createElement("li");
  item.className = role;
  item.textContent = `${role === "assistant" ? "ClinIQ" : "You"}: ${text}`;
  transcript.append(item);
  transcript.scrollTop = transcript.scrollHeight;
}

// Stops the animation-frame loop that samples microphone volume.
function stopSpeechDetection() {
  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = undefined;
  }
}

// Converts analyser waveform samples to an RMS volume value for voice detection.
function getAudioLevel() {
  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);
  let total = 0;
  for (const sample of samples) {
    const value = (sample - 128) / 128;
    total += value * value;
  }
  return Math.sqrt(total / samples.length);
}

// Starts a new MediaRecorder only after the analyser detects real speech.
function startTurnRecording() {
  recordedChunks = [];
  recorder = new MediaRecorder(microphoneStream);
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  });
  recorder.addEventListener("stop", handleRecordedTurn, { once: true });
  recorder.start();
  speechStartedAt = performance.now();
  silenceStartedAt = undefined;
  setStatus(microphoneStatus, "Listening to you...");
  setStatus(assistantStatus, "Waiting for your turn");
}

// Watches microphone volume and ends the current turn after sustained silence.
function waitForSpeech() {
  if (!callActive || muted || processing || assistantSpeaking || !analyser) return;

  const level = getAudioLevel();
  const now = performance.now();
  if (level >= SPEECH_DETECTION_THRESHOLD) {
    if (recorder?.state !== "recording") startTurnRecording();
    silenceStartedAt = undefined;
  } else if (recorder?.state === "recording") {
    if (!silenceStartedAt) silenceStartedAt = now;
    if (now - silenceStartedAt >= SILENCE_DURATION_MS) {
      stopSpeechDetection();
      recorder.stop();
      return;
    }
  }

  animationFrame = requestAnimationFrame(waitForSpeech);
}

// Returns the active call to passive listening after processing or playback.
function beginListening() {
  if (!callActive || muted || processing || assistantSpeaking) return;
  stopSpeechDetection();
  silenceStartedAt = undefined;
  setStatus(microphoneStatus, "Listening...");
  setStatus(assistantStatus, "Listening again...");
  animationFrame = requestAnimationFrame(waitForSpeech);
}

// Rewords rate-limit failures so the user knows the call remains available.
function formatVoiceError(message) {
  if (/\b429\b|rate limit|quota/i.test(message)) {
    return "Groq rate limit or quota reached. The call will keep listening.";
  }
  return `Voice Call error: ${message}`;
}

// Validates one recorded turn, sends it to the Groq pipeline, and preserves history.
async function handleRecordedTurn() {
  const recordingDuration = performance.now() - speechStartedAt;
  const mimeType = recorder?.mimeType || "audio/webm";
  const audioBlob = new Blob(recordedChunks, { type: mimeType });
  recorder = undefined;

  if (!callActive || muted) return;
  if (recordingDuration < MIN_RECORDING_DURATION_MS || audioBlob.size < MIN_RECORDING_BYTES) {
    beginListening();
    return;
  }

  processing = true;
  setStatus(microphoneStatus, "Processing");
  setStatus(assistantStatus, "Thinking...");
  try {
    const form = new FormData();
    form.append("audio", audioBlob, "voice-turn.webm");
    form.append("history_json", JSON.stringify(history));
    const response = await fetch("/api/ai/voice/turn", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The voice turn could not be completed.");

    addTranscript("user", payload.userText);
    history.push({ role: "user", content: payload.userText });
    addTranscript("assistant", payload.assistantText);
    history.push({ role: "assistant", content: payload.assistantText });
    await playAssistantAudio(payload.audioBase64);
  } catch (error) {
    reportError(formatVoiceError(error.message));
  } finally {
    processing = false;
    if (callActive && !muted && !assistantSpeaking) beginListening();
  }
}

// Plays the returned WAV and keeps microphone detection paused to avoid feedback.
function playAssistantAudio(audioBase64) {
  return new Promise((resolve) => {
    assistantSpeaking = true;
    stopSpeechDetection();
    setStatus(microphoneStatus, "Paused");
    setStatus(assistantStatus, "Assistant speaking...");
    currentAudio = new Audio(`data:audio/wav;base64,${audioBase64}`);

    // Restores the call state after successful playback or an audio playback error.
    const finishPlayback = () => {
      currentAudio?.removeEventListener("ended", finishPlayback);
      currentAudio?.removeEventListener("error", finishPlayback);
      currentAudio = undefined;
      assistantSpeaking = false;
      resolve();
    };

    currentAudio.addEventListener("ended", finishPlayback, { once: true });
    currentAudio.addEventListener("error", finishPlayback, { once: true });
    currentAudio.play().catch(() => {
      reportError("Unable to play the assistant audio.");
      finishPlayback();
    });
  });
}

// Acquires the microphone, initializes the analyser, and starts automatic listening.
async function startCall() {
  reportError("");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    reportError("This browser does not support microphone recording for Voice Call.");
    return;
  }

  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    audioContext = new AudioContext();
    await audioContext.resume();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = ANALYSER_FFT_SIZE;
    audioContext.createMediaStreamSource(microphoneStream).connect(analyser);

    callActive = true;
    muted = false;
    startCallButton.disabled = true;
    muteButton.disabled = false;
    muteButton.textContent = "Mute";
    muteButton.setAttribute("aria-pressed", "false");
    endCallButton.disabled = false;
    setStatus(connectionStatus, "Call active");
    beginListening();
  } catch (error) {
    setStatus(connectionStatus, "Unavailable");
    setStatus(microphoneStatus, "Denied");
    reportError("Microphone access is required to start a Voice Call. Allow it in your browser and try again.");
  }
}

// Enables or disables microphone tracks while preserving the active call session.
function toggleMute() {
  if (!callActive || !microphoneStream) return;
  muted = !muted;
  microphoneStream.getAudioTracks().forEach((track) => {
    track.enabled = !muted;
  });
  muteButton.textContent = muted ? "Unmute" : "Mute";
  muteButton.setAttribute("aria-pressed", String(muted));

  if (muted) {
    stopSpeechDetection();
    if (recorder?.state === "recording") recorder.stop();
    setStatus(microphoneStatus, "Muted");
    setStatus(assistantStatus, "Waiting");
  } else {
    beginListening();
  }
}

// Stops recording, playback, microphone tracks, and all automatic call activity.
function endCall() {
  callActive = false;
  processing = false;
  assistantSpeaking = false;
  stopSpeechDetection();
  if (recorder?.state === "recording") recorder.stop();
  recorder = undefined;
  currentAudio?.pause();
  currentAudio = undefined;
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = undefined;
  analyser = undefined;
  audioContext?.close();
  audioContext = undefined;
  setStatus(connectionStatus, "Call ended");
  setStatus(microphoneStatus, "Off");
  setStatus(assistantStatus, "Waiting");
  startCallButton.disabled = false;
  muteButton.disabled = true;
  muteButton.textContent = "Mute";
  muteButton.setAttribute("aria-pressed", "false");
  endCallButton.disabled = true;
}

// Wire the three call-level controls; individual turns never need a button press.
startCallButton.addEventListener("click", startCall);
muteButton.addEventListener("click", toggleMute);
endCallButton.addEventListener("click", endCall);
