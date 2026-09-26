/**
 * HealHub Voice Emotion Demo - Frontend Logic
 * Modern ES6 JavaScript handling live Web Audio recording, drag-and-drop uploads,
 * API communication, dynamic emotion badge styling, and responsive UX.
 */

let activeTab = 'record';
let currentAudioBlob = null;
let currentAudioFile = null;

// Audio Recorder State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordStartTime = 0;
let timerInterval = null;

// DOM Elements
const tabRecordBtn = document.getElementById('tabRecordBtn');
const tabUploadBtn = document.getElementById('tabUploadBtn');
const recordContent = document.getElementById('recordContent');
const uploadContent = document.getElementById('uploadContent');

const visualizerContainer = document.getElementById('visualizerContainer');
const recordTimer = document.getElementById('recordTimer');
const recorderStatus = document.getElementById('recorderStatus');
const recordToggleBtn = document.getElementById('recordToggleBtn');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

const audioPreviewWrapper = document.getElementById('audioPreviewWrapper');
const audioFileName = document.getElementById('audioFileName');
const audioPlayer = document.getElementById('audioPlayer');

const analyzeBtn = document.getElementById('analyzeBtn');
const btnSpinner = document.getElementById('btnSpinner');

const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const resultsContent = document.getElementById('resultsContent');
const executionTimeBadge = document.getElementById('executionTimeBadge');

const transcriptOutput = document.getElementById('transcriptOutput');
const emotionBadge = document.getElementById('emotionBadge');
const emotionMessage = document.getElementById('emotionMessage');
const supportAlert = document.getElementById('supportAlert');
const supportMessageText = document.getElementById('supportMessageText');

// Setup Drag & Drop Handlers on Load
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
});

// Tab Switcher
function switchTab(tab) {
    activeTab = tab;
    if (tab === 'record') {
        tabRecordBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        recordContent.classList.add('active');
        uploadContent.classList.remove('active');
    } else {
        tabUploadBtn.classList.add('active');
        tabRecordBtn.classList.remove('active');
        uploadContent.classList.add('active');
        recordContent.classList.remove('active');
    }
}

// Drag & Drop Setup
function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            processSelectedFile(files[0]);
        }
    });
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files && files.length > 0) {
        processSelectedFile(files[0]);
    }
}

function processSelectedFile(file) {
    if (!file.type.startsWith('audio/')) {
        alert('Please select a valid audio file (WAV, MP3, M4A, OGG, WEBM).');
        return;
    }
    
    currentAudioFile = file;
    currentAudioBlob = file;
    audioFileName.textContent = file.name;
    
    const audioUrl = URL.createObjectURL(file);
    audioPlayer.src = audioUrl;
    audioPreviewWrapper.classList.remove('hidden');
    analyzeBtn.disabled = false;
}

// Live Audio Recorder Logic
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioChunks = [];
        let mimeType = 'audio/webm';
        if (!MediaRecorder.isTypeSupported('audio/webm')) {
            if (MediaRecorder.isTypeSupported('audio/mp4')) mimeType = 'audio/mp4';
            else if (MediaRecorder.isTypeSupported('audio/ogg')) mimeType = 'audio/ogg';
            else mimeType = '';
        }
        
        mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blobType = mediaRecorder.mimeType || 'audio/webm';
            currentAudioBlob = new Blob(audioChunks, { type: blobType });
            currentAudioFile = new File([currentAudioBlob], "recorded_voice_note.webm", { type: blobType });
            
            const audioUrl = URL.createObjectURL(currentAudioBlob);
            audioPlayer.src = audioUrl;
            audioFileName.textContent = "Live Recorded Voice Note";
            audioPreviewWrapper.classList.remove('hidden');
            analyzeBtn.disabled = false;
            
            // Stop all tracks to release mic
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        recordStartTime = Date.now();
        
        visualizerContainer.classList.add('recording');
        recordToggleBtn.textContent = 'Stop Recording';
        recordToggleBtn.classList.remove('btn-primary');
        recordToggleBtn.classList.add('btn-action');
        recorderStatus.textContent = 'Recording in progress... Speak clearly';

        timerInterval = setInterval(updateTimer, 1000);
    } catch (err) {
        alert('Microphone access denied or not supported in your browser: ' + err.message);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        clearInterval(timerInterval);
        
        visualizerContainer.classList.remove('recording');
        recordToggleBtn.textContent = 'Start Recording';
        recordToggleBtn.classList.remove('btn-action');
        recordToggleBtn.classList.add('btn-primary');
        recorderStatus.textContent = 'Recording saved. Tap Analyze Voice Note to evaluate.';
    }
}

function updateTimer() {
    const elapsedSeconds = Math.floor((Date.now() - recordStartTime) / 1000);
    const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
    const secs = String(elapsedSeconds % 60).padStart(2, '0');
    recordTimer.textContent = `${mins}:${secs}`;
}

function clearAudio() {
    currentAudioBlob = null;
    currentAudioFile = null;
    audioPlayer.src = '';
    audioPreviewWrapper.classList.add('hidden');
    analyzeBtn.disabled = true;
    recordTimer.textContent = '00:00';
    recorderStatus.textContent = 'Tap Start to begin recording';
    if (fileInput) fileInput.value = '';
}

// Analyze Voice Note API Call
async function analyzeVoice() {
    if (!currentAudioFile && !currentAudioBlob) {
        alert('Please record or upload an audio file first.');
        return;
    }

    // Set Loading State
    analyzeBtn.disabled = true;
    btnSpinner.classList.remove('hidden');
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing Voice...';

    try {
        // Convert any audio format (WebM, MP3, M4A, OGG) to 16kHz mono WAV in browser
        let fileToSend = currentAudioFile;
        try {
            const blobToConvert = currentAudioBlob || currentAudioFile;
            const wavBlob = await convertBlobToWav(blobToConvert);
            fileToSend = new File([wavBlob], "voice_note.wav", { type: "audio/wav" });
        } catch (convErr) {
            console.warn("WAV normalization fallback:", convErr);
        }

        const formData = new FormData();
        formData.append('file', fileToSend);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert('Failed to analyze voice note: ' + err.message);
    } finally {
        analyzeBtn.disabled = false;
        btnSpinner.classList.add('hidden');
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Voice Note';
    }
}

// Browser Web Audio API AudioBuffer -> 16kHz mono WAV Blob Converter
async function convertBlobToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    
    const targetSampleRate = 16000;
    const offlineCtx = new OfflineAudioContext(
        1,
        Math.ceil(decodedBuffer.duration * targetSampleRate),
        targetSampleRate
    );
    
    const source = offlineCtx.createBufferSource();
    source.buffer = decodedBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);
    
    const renderedBuffer = await offlineCtx.startRendering();
    const pcmData = renderedBuffer.getChannelData(0);
    
    const wavBuffer = encodeWAV(pcmData, targetSampleRate);
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
}

// Render Results on Screen
function renderResults(data) {
    if (!data.success) {
        alert(data.error || 'Voice analysis failed.');
        return;
    }

    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    // Execution time badge
    if (data.execution_time_seconds !== undefined) {
        executionTimeBadge.textContent = `Analyzed in ${data.execution_time_seconds}s`;
        executionTimeBadge.classList.remove('hidden');
    }

    // Transcript
    transcriptOutput.textContent = data.transcript || "No speech detected in audio.";

    // Emotion Badge
    const rawLabel = (data.emotion_label || 'neutral').toLowerCase();
    emotionBadge.className = `emotion-badge ${rawLabel}`;
    emotionBadge.textContent = rawLabel.charAt(0).toUpperCase() + rawLabel.slice(1);

    // Emotion Explanation
    emotionMessage.textContent = data.emotion || "Detected emotional tone analysis completed.";

    // Support Alert
    if (data.support_message) {
        supportMessageText.textContent = data.support_message;
        supportAlert.classList.remove('hidden');
    } else {
        supportAlert.classList.add('hidden');
    }
}

// Copy Transcript to Clipboard
function copyTranscript() {
    const text = transcriptOutput.textContent;
    if (text) {
        navigator.clipboard.writeText(text).then(() => {
            const copyBtn = document.querySelector('.btn-copy');
            const origHtml = copyBtn.innerHTML;
            copyBtn.innerHTML = 'Copied!';
            setTimeout(() => {
                copyBtn.innerHTML = origHtml;
            }, 2000);
        }).catch(err => {
            console.error('Clipboard copy failed:', err);
        });
    }
}
