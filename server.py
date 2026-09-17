import os
import sys
import cv2
import time
import uuid
import numpy as np
import base64
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, SAMPLE_RATE, VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT
from pipeline.video_pipeline import VideoPipeline
from pipeline.audio_pipeline import AudioPipeline
from models.lipsync_detector import LipSyncConsistency
from models.forensic_features import ForensicFeatureExtractor
from models.audio_dsp_features import AudioDSPFeatureExtractor

# Global models
video_pipeline = None
audio_pipeline = None
lipsync_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_pipeline, audio_pipeline, lipsync_detector
    print("[Server] Initializing pipelines...")
    video_pipeline = VideoPipeline(device=DEVICE)
    audio_pipeline = AudioPipeline(device=DEVICE)
    # Roadmap item 1 — shared across /ws/video and /ws/audio so both streams
    # feed the same rolling correlation window. Attaching it to the
    # pipelines is opt-in (see VideoPipeline/AudioPipeline.lipsync) and
    # doesn't change process_frame()'s signature or return shape.
    lipsync_detector = LipSyncConsistency()
    video_pipeline.lipsync = lipsync_detector
    audio_pipeline.lipsync = lipsync_detector
    # Roadmap item 2 — fuses residual/LBP forensic cues into the ViT's own
    # fake_probability before it reaches alert_system.py / the WebSocket
    # response. Opt-in (see VideoPipeline.forensics); ~3-4ms/detection at
    # 224x224 after downscaling the LBP step, see CLAUDE_CODE_HANDOFF.md §6b.
    video_pipeline.forensics = ForensicFeatureExtractor()
    # Roadmap item 3 — jitter/shimmer DSP cues fused into Wav2Vec2's own
    # fake_probability, same opt-in pattern (see AudioPipeline.audio_dsp).
    # ~12ms/window on this machine, well inside the 1s audio stride budget.
    audio_pipeline.audio_dsp = AudioDSPFeatureExtractor()
    print("[Server] Pipelines ready.")
    yield
    print("[Server] Shutting down...")

app = FastAPI(title="Deepfake Detector API", lifespan=lifespan)

# Setup static files and uploads directory
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
    
    file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    results = []
    fake_count = 0
    frame_count = 0

    try:
        # process_uploaded_video yields frames
        for annotated, result in video_pipeline.process_uploaded_video(file_path):
            frame_count += 1
            if result.get("label") == "FAKE":
                fake_count += 1
            results.append(result)
            
        total_analyzed = len([r for r in results if r.get("label") in ["REAL", "FAKE"]])
        if total_analyzed == 0:
            return {"error": "No faces detected in the video."}
            
        fake_pct = fake_count / total_analyzed * 100
        avg_fake_prob = np.mean([r.get("fake_probability", 0) for r in results])
        
        verdict = "AUTHENTIC"
        if fake_pct > 50:
            verdict = "FAKE"
        elif fake_pct > 20:
            verdict = "SUSPICIOUS"
            
        return {
            "total_frames": total_analyzed,
            "fake_frames": fake_count,
            "fake_percentage": float(fake_pct),
            "average_fake_probability": float(avg_fake_prob),
            "verdict": verdict
        }
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
        
    file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        results = audio_pipeline.process_audio_file(file_path)
        if not results:
            return {"error": "Could not analyze audio file. It may be too short."}
            
        avg_fake_prob = float(np.mean([r["fake_probability"] for r in results]))
        fake_windows = sum(1 for r in results if r["label"] == "FAKE")
        
        verdict = "FAKE" if avg_fake_prob > 0.5 else "AUTHENTIC"
        
        return {
            "total_windows": len(results),
            "fake_windows": fake_windows,
            "average_fake_probability": avg_fake_prob,
            "verdict": verdict,
            "details": results
        }
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting base64 image data
            # Format: data:image/jpeg;base64,....
            if "base64," in data:
                base64_str = data.split("base64,")[1]
            else:
                base64_str = data
                
            img_data = base64.b64decode(base64_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # OpenCV decodes as BGR, but Gradio/Canvas sends RGB encoded.
            # Depending on browser encoding, we might need to swap channels.
            # Usually HTML canvas toDataUrl('image/jpeg') decodes directly well with OpenCV.
            # We'll just resize it.
            frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
            annotated, result = video_pipeline.process_frame(frame)
            lipsync_result = lipsync_detector.get_sync_score()

            # Encode annotated frame back to base64
            _, buffer = cv2.imencode('.jpg', annotated)
            out_base64 = base64.b64encode(buffer).decode('utf-8')

            response = {
                "image": f"data:image/jpeg;base64,{out_base64}",
                "result": {
                    "label": result.get("label", "DETECTING..."),
                    "confidence": float(result.get("confidence", 0.0)),
                    "fake_probability": float(result.get("fake_probability", 0.0))
                },
                "lipsync": {
                    "label": lipsync_result.get("label", "INSUFFICIENT_DATA"),
                    "sync_score": float(lipsync_result.get("sync_score", 0.0)),
                },
                "fps": float(video_pipeline.fps)
            }
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("[Video WS] Client disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Video WS] Error: {e}")

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = np.zeros(0, dtype=np.float32)
    min_samples = int(3.0 * SAMPLE_RATE)  # From config AUDIO_WINDOW_SECONDS
    
    try:
        while True:
            # Receive raw float32 array from client
            # The client should send Float32Array
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            lipsync_detector.update_audio(chunk, SAMPLE_RATE)

            audio_buffer = np.concatenate([audio_buffer, chunk])
            
            # Process if we have enough data (e.g. 3 seconds)
            if len(audio_buffer) >= min_samples:
                window = audio_buffer[:min_samples]
                
                # Normalize exactly like in app.py process_audio_chunk
                max_val = np.max(np.abs(window))
                if max_val > 0:
                    window = window / max_val
                    
                result = audio_pipeline.detector.predict(window)
                if audio_pipeline.audio_dsp is not None:
                    result = audio_pipeline.audio_dsp.fuse(result, window, SAMPLE_RATE)

                # Slide the buffer (1 second stride based on AUDIO_STRIDE_SECONDS in config)
                stride = int(1.0 * SAMPLE_RATE)
                audio_buffer = audio_buffer[stride:]
                
                response = {
                    "label": result.get("label", "REAL"),
                    "fake_probability": float(result.get("fake_probability", 0.0))
                }
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        print("[Audio WS] Client disconnected")
    except Exception as e:
        print(f"[Audio WS] Error: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
