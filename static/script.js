// ==========================================
// Tabbing Navigation
// ==========================================
const tabs = document.querySelectorAll('.nav-links li');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
    });
});

// ==========================================
// Video Streaming (WebSocket)
// ==========================================
const startVideoBtn = document.getElementById('start-video-btn');
const videoElement = document.getElementById('webcam-video');
const canvasElement = document.getElementById('webcam-canvas');
const processedVideo = document.getElementById('processed-video');
const videoPlaceholder = document.getElementById('video-placeholder');

const vidLabel = document.getElementById('video-label');
const vidConf = document.getElementById('video-confidence');
const vidFps = document.getElementById('video-fps');

let videoWs = null;
let videoStream = null;
let videoInterval = null;
let isVideoRunning = false;
let waitingForResponse = false;

startVideoBtn.addEventListener('click', async () => {
    if (isVideoRunning) {
        stopVideo();
    } else {
        await startVideo();
    }
});

async function startVideo() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = videoStream;

        // IMPORTANT: The video element must NOT be display:none for the browser
        // to render frames. We make it invisible but still rendered.
        videoElement.style.position = 'absolute';
        videoElement.style.width = '1px';
        videoElement.style.height = '1px';
        videoElement.style.opacity = '0';
        videoElement.style.pointerEvents = 'none';
        videoElement.style.overflow = 'hidden';
        videoElement.classList.remove('hidden');

        // Wait for the video to actually start playing
        await videoElement.play();

        videoWs = new WebSocket(`ws://${window.location.host}/ws/video`);

        videoWs.onopen = () => {
            console.log("[Video] WebSocket connected");
            isVideoRunning = true;
            waitingForResponse = false;
            startVideoBtn.classList.add('active');
            videoPlaceholder.classList.add('hidden');
            processedVideo.classList.remove('hidden');

            const ctx = canvasElement.getContext('2d');

            // Send frames at ~5 FPS (200ms interval) to avoid overwhelming
            videoInterval = setInterval(() => {
                if (!isVideoRunning || !videoWs || videoWs.readyState !== WebSocket.OPEN) return;
                // Throttle: don't send a new frame until the previous response arrived
                if (waitingForResponse) return;

                if (videoElement.readyState >= 2 && videoElement.videoWidth > 0) {
                    if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
                        canvasElement.width = videoElement.videoWidth;
                        canvasElement.height = videoElement.videoHeight;
                    }
                    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

                    const base64Data = canvasElement.toDataURL('image/jpeg', 0.7);
                    videoWs.send(base64Data);
                    waitingForResponse = true;
                }
            }, 200);
        };

        videoWs.onmessage = (event) => {
            waitingForResponse = false;
            try {
                const data = JSON.parse(event.data);
                processedVideo.src = data.image;

                vidLabel.textContent = data.result.label;
                vidConf.textContent = Math.round(data.result.confidence * 100) + '%';
                vidFps.textContent = Math.round(data.fps);

                if (data.result.label === 'FAKE') {
                    vidLabel.className = 'value fake';
                } else if (data.result.label === 'REAL') {
                    vidLabel.className = 'value real';
                } else {
                    vidLabel.className = 'value neutral';
                }
            } catch (e) {
                console.error("[Video] Error parsing response:", e);
            }
        };

        videoWs.onerror = (e) => console.error("[Video] WS error", e);

        // Don't call stopVideo from onclose to avoid circular cleanup
        videoWs.onclose = () => {
            console.log("[Video] WebSocket closed");
            if (isVideoRunning) {
                // Only cleanup if user didn't manually stop
                cleanupVideo();
            }
        };

    } catch (e) {
        console.error("Camera access denied or error", e);
        alert("Camera access failed. Please ensure permissions are granted.");
    }
}

function cleanupVideo() {
    isVideoRunning = false;
    waitingForResponse = false;
    startVideoBtn.classList.remove('active');
    videoPlaceholder.classList.remove('hidden');
    processedVideo.classList.add('hidden');

    vidLabel.textContent = 'STANDBY';
    vidLabel.className = 'value neutral';
    vidConf.textContent = '0%';
    vidFps.textContent = '0';

    if (videoInterval) {
        clearInterval(videoInterval);
        videoInterval = null;
    }
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    // Hide video element again
    videoElement.classList.add('hidden');
    videoElement.style = '';
}

function stopVideo() {
    isVideoRunning = false;
    waitingForResponse = false;

    if (videoInterval) {
        clearInterval(videoInterval);
        videoInterval = null;
    }

    // Close websocket FIRST (this won't recurse because isVideoRunning is false)
    if (videoWs) {
        const ws = videoWs;
        videoWs = null;
        ws.close();
    }

    // Then cleanup
    cleanupVideo();
}

// ==========================================
// Audio Streaming (WebSocket)
// ==========================================
const startAudioBtn = document.getElementById('start-audio-btn');
const audioPlaceholder = document.getElementById('audio-placeholder');
const audioLabel = document.getElementById('audio-label');
const audioProb = document.getElementById('audio-prob');

let audioWs = null;
let audioStream = null;
let audioContext = null;
let processor = null;
let isAudioRunning = false;

startAudioBtn.addEventListener('click', async () => {
    if (isAudioRunning) {
        stopAudio();
    } else {
        await startAudio();
    }
});

async function startAudio() {
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Force 16kHz for backend using AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(audioStream);

        // Use ScriptProcessor for raw data access
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        audioWs = new WebSocket(`ws://${window.location.host}/ws/audio`);
        audioWs.binaryType = 'arraybuffer';

        audioWs.onopen = () => {
            console.log("[Audio] WebSocket connected");
            isAudioRunning = true;
            startAudioBtn.classList.add('active');
            audioPlaceholder.querySelector('span').textContent = "Listening...";
            audioPlaceholder.querySelector('i').className = "ph ph-microphone";

            processor.onaudioprocess = (e) => {
                if (audioWs && audioWs.readyState === WebSocket.OPEN) {
                    const audioData = e.inputBuffer.getChannelData(0);
                    // IMPORTANT: Must send the underlying ArrayBuffer, not the Float32Array view
                    const buffer = new Float32Array(audioData).buffer;
                    audioWs.send(buffer);
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
        };

        audioWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                audioLabel.textContent = data.label;
                audioProb.textContent = Math.round(data.fake_probability * 100) + '%';
                audioLabel.className = 'value ' + (data.label === 'FAKE' ? 'fake' : 'real');
            } catch (e) {
                console.error("[Audio] Error parsing response:", e);
            }
        };

        audioWs.onerror = (e) => console.error("[Audio] WS error", e);
        audioWs.onclose = () => {
            console.log("[Audio] WebSocket closed");
            if (isAudioRunning) {
                cleanupAudio();
            }
        };

    } catch (e) {
        console.error("Audio access denied or error", e);
        alert("Microphone access failed. Please ensure permissions are granted.");
    }
}

function cleanupAudio() {
    isAudioRunning = false;
    startAudioBtn.classList.remove('active');
    audioPlaceholder.querySelector('span').textContent = "Microphone inactive";
    audioPlaceholder.querySelector('i').className = "ph ph-microphone-slash";

    audioLabel.textContent = 'STANDBY';
    audioLabel.className = 'value neutral';
    audioProb.textContent = '0%';

    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}

function stopAudio() {
    isAudioRunning = false;

    if (processor) {
        processor.disconnect();
        processor = null;
    }

    if (audioWs) {
        const ws = audioWs;
        audioWs = null;
        ws.close();
    }

    cleanupAudio();
}


// ==========================================
// Screen Detection (getDisplayMedia)
// ==========================================
const startScreenBtn = document.getElementById('start-screen-btn');
const screenVideo = document.getElementById('screen-video');
const screenCanvas = document.getElementById('screen-canvas');
const processedScreen = document.getElementById('processed-screen');
const screenPlaceholder = document.getElementById('screen-placeholder');

const scrLabel = document.getElementById('screen-label');
const scrConf = document.getElementById('screen-confidence');
const scrFps = document.getElementById('screen-fps');
const scrProb = document.getElementById('screen-prob');

let screenWs = null;
let screenStream = null;
let screenInterval = null;
let isScreenRunning = false;
let screenWaitingForResponse = false;

startScreenBtn.addEventListener('click', async () => {
    if (isScreenRunning) {
        stopScreen();
    } else {
        await startScreen();
    }
});

async function startScreen() {
    try {
        screenStream = await navigator.mediaDevices.getDisplayMedia({
            video: { cursor: "always" },
            audio: false
        });

        screenVideo.srcObject = screenStream;

        // Make video element invisible but still rendered (so canvas can draw frames)
        screenVideo.style.position = 'absolute';
        screenVideo.style.width = '1px';
        screenVideo.style.height = '1px';
        screenVideo.style.opacity = '0';
        screenVideo.style.pointerEvents = 'none';
        screenVideo.style.overflow = 'hidden';
        screenVideo.classList.remove('hidden');

        await screenVideo.play();

        // Detect when user stops sharing via browser UI
        screenStream.getVideoTracks()[0].onended = () => {
            stopScreen();
        };

        screenWs = new WebSocket(`ws://${window.location.host}/ws/video`);

        screenWs.onopen = () => {
            console.log("[Screen] WebSocket connected");
            isScreenRunning = true;
            screenWaitingForResponse = false;
            startScreenBtn.classList.add('active');
            screenPlaceholder.classList.add('hidden');
            processedScreen.classList.remove('hidden');

            const ctx = screenCanvas.getContext('2d');

            // Send frames at ~5 FPS
            screenInterval = setInterval(() => {
                if (!isScreenRunning || !screenWs || screenWs.readyState !== WebSocket.OPEN) return;
                if (screenWaitingForResponse) return;

                if (screenVideo.readyState >= 2 && screenVideo.videoWidth > 0) {
                    if (screenCanvas.width !== screenVideo.videoWidth || screenCanvas.height !== screenVideo.videoHeight) {
                        screenCanvas.width = screenVideo.videoWidth;
                        screenCanvas.height = screenVideo.videoHeight;
                    }
                    ctx.drawImage(screenVideo, 0, 0, screenCanvas.width, screenCanvas.height);

                    const base64Data = screenCanvas.toDataURL('image/jpeg', 0.7);
                    screenWs.send(base64Data);
                    screenWaitingForResponse = true;
                }
            }, 200);
        };

        screenWs.onmessage = (event) => {
            screenWaitingForResponse = false;
            try {
                const data = JSON.parse(event.data);
                processedScreen.src = data.image;

                scrLabel.textContent = data.result.label;
                scrConf.textContent = Math.round(data.result.confidence * 100) + '%';
                scrFps.textContent = Math.round(data.fps);
                scrProb.textContent = Math.round(data.result.fake_probability * 100) + '%';

                if (data.result.label === 'FAKE') {
                    scrLabel.className = 'value fake';
                } else if (data.result.label === 'REAL') {
                    scrLabel.className = 'value real';
                } else {
                    scrLabel.className = 'value neutral';
                }
            } catch (e) {
                console.error("[Screen] Error parsing response:", e);
            }
        };

        screenWs.onerror = (e) => console.error("[Screen] WS error", e);
        screenWs.onclose = () => {
            console.log("[Screen] WebSocket closed");
            if (isScreenRunning) {
                cleanupScreen();
            }
        };

    } catch (e) {
        console.error("Screen share denied or error", e);
        // User cancelled the screen share picker — not an error
        if (e.name !== 'NotAllowedError') {
            alert("Screen share failed. Please try again.");
        }
    }
}

function cleanupScreen() {
    isScreenRunning = false;
    screenWaitingForResponse = false;
    startScreenBtn.classList.remove('active');
    screenPlaceholder.classList.remove('hidden');
    processedScreen.classList.add('hidden');

    scrLabel.textContent = 'STANDBY';
    scrLabel.className = 'value neutral';
    scrConf.textContent = '0%';
    scrFps.textContent = '0';
    scrProb.textContent = '0%';

    if (screenInterval) {
        clearInterval(screenInterval);
        screenInterval = null;
    }
    if (screenStream) {
        screenStream.getTracks().forEach(track => track.stop());
        screenStream = null;
    }
    screenVideo.classList.add('hidden');
    screenVideo.style = '';
}

function stopScreen() {
    isScreenRunning = false;
    screenWaitingForResponse = false;

    if (screenInterval) {
        clearInterval(screenInterval);
        screenInterval = null;
    }

    if (screenWs) {
        const ws = screenWs;
        screenWs = null;
        ws.close();
    }

    cleanupScreen();
}

function setupUpload(inputId, dropId, resultsId, endpoint) {
    const input = document.getElementById(inputId);
    const dropZone = document.getElementById(dropId);
    const resultsBox = document.getElementById(resultsId);

    // Browse click
    dropZone.querySelector('.browse-link').addEventListener('click', () => input.click());

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleFile(input.files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length) {
            handleFile(input.files[0]);
        }
    });

    async function handleFile(file) {
        dropZone.innerHTML = `<div class="loader"></div><p style="margin-top:1rem">Analyzing ${file.name}...</p>`;
        resultsBox.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            // Restore Drop Zone
            dropZone.innerHTML = `<i class="ph ph-check-circle"></i><p>Analysis Complete</p><p style="font-size:0.8rem;text-decoration:underline;cursor:pointer" onclick="document.getElementById('${inputId}').click()">Analyze another</p>`;

            renderResults(data, resultsBox);

        } catch (e) {
            alert("Upload failed. Ensure backend is running.");
            dropZone.innerHTML = `<i class="ph ph-warning-circle" style="color:var(--danger)"></i><p>Upload Failed. Try again.</p>`;
        }
    }
}

function renderResults(data, container) {
    if (data.error) {
        container.innerHTML = `<h4 style="color:var(--danger)">Analysis Error</h4><p>${data.error}</p>`;
        container.classList.remove('hidden');
        return;
    }

    let html = `<h4>Analysis Report</h4>`;

    for (const [key, val] of Object.entries(data)) {
        if (key === 'details') continue;

        let displayVal = val;
        if (typeof val === 'number' && key.includes('prob')) displayVal = Math.round(val * 100) + '%';
        if (typeof val === 'number' && key.includes('percentage')) displayVal = val.toFixed(1) + '%';

        const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        if (key !== 'verdict') {
            html += `<div class="result-row"><span class="key">${displayKey}</span><span class="val">${displayVal}</span></div>`;
        }
    }

    if (data.verdict) {
        const isFake = data.verdict === 'FAKE' || data.verdict === 'SUSPICIOUS';
        const cssClass = isFake ? 'fake' : 'real';
        const emoji = isFake ? '⚠️' : '✅';
        html += `<div class="verdict-banner ${cssClass}">${emoji} ${data.verdict}</div>`;
    }

    container.innerHTML = html;
    container.classList.remove('hidden');
}

setupUpload('video-upload-input', 'video-drop-zone', 'video-upload-results', '/api/analyze/video');
setupUpload('audio-upload-input', 'audio-drop-zone', 'audio-upload-results', '/api/analyze/audio');
