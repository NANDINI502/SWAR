"""
Face Extractor — Detects and crops faces from video frames using MTCNN.
Optimized for real-time performance with configurable frame skipping.
"""

import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN


from config import DEVICE, FACE_INPUT_SIZE, FACE_DETECTION_THRESHOLD


class FaceExtractor:
    """Extracts aligned face crops from video frames using MTCNN."""

    def __init__(self, device: str = DEVICE, min_face_size: int = 60):
        self.device = device
        self.mtcnn = MTCNN(
            image_size=FACE_INPUT_SIZE,
            margin=20,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, FACE_DETECTION_THRESHOLD],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=True,  # Detect all faces in frame
        )
        print(f"[FaceExtractor] Initialized on {device}")

    def extract_faces(self, frame: np.ndarray):
        """
        Detect and extract face crops from a BGR frame.

        Args:
            frame: BGR numpy array from OpenCV (H, W, 3)

        Returns:
            List of dicts, each containing:
                - 'face': numpy array (224, 224, 3), RGB uint8, UNnormalized
                  (VideoDetectorInference applies its own model-specific
                  preprocessing — don't normalize here)
                - 'box': (x1, y1, x2, y2) bounding box coordinates
                - 'confidence': detection confidence score
        """
        # Convert BGR → RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Detect faces
        boxes, confidences = self.mtcnn.detect(pil_image)

        results = []
        if boxes is not None:
            for box, conf in zip(boxes, confidences):
                if conf is None or conf < FACE_DETECTION_THRESHOLD:
                    continue

                # Clamp box coordinates to frame bounds
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(frame.shape[1], int(box[2]))
                y2 = min(frame.shape[0], int(box[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop and resize (kept as raw RGB uint8 — the detector's own
                # HuggingFace processor handles resizing/normalization to
                # match however its checkpoint was actually trained)
                face_crop = rgb_frame[y1:y2, x1:x2]
                face_resized = cv2.resize(face_crop, (FACE_INPUT_SIZE, FACE_INPUT_SIZE))

                results.append({
                    "face": face_resized,
                    "box": (x1, y1, x2, y2),
                    "confidence": float(conf),
                })

        return results

    def extract_faces_with_landmarks(self, frame: np.ndarray):
        """
        Same as extract_faces(), but each result also carries a 'landmarks'
        dict (left_eye, right_eye, nose, mouth_left, mouth_right — pixel
        coordinates in `frame`'s coordinate space, i.e. NOT relative to the
        resized 'face' crop). Used by models/lipsync_detector.py.

        Kept as a separate method (rather than adding 'landmarks' to
        extract_faces()'s return dicts) so extract_faces()'s existing return
        shape — and everything already depending on it — doesn't change.
        The landmarks are a free byproduct of the same MTCNN forward pass
        (ONet already regresses them internally), so this costs the same as
        extract_faces() — not a second model or a second detection pass.

        Returns:
            List of dicts: 'face', 'box', 'confidence' (same as
            extract_faces()) plus 'landmarks': {
                'left_eye': (x, y), 'right_eye': (x, y), 'nose': (x, y),
                'mouth_left': (x, y), 'mouth_right': (x, y)
            }
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        boxes, confidences, landmarks = self.mtcnn.detect(pil_image, landmarks=True)

        results = []
        if boxes is not None:
            for box, conf, lms in zip(boxes, confidences, landmarks):
                if conf is None or conf < FACE_DETECTION_THRESHOLD:
                    continue

                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(frame.shape[1], int(box[2]))
                y2 = min(frame.shape[0], int(box[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = rgb_frame[y1:y2, x1:x2]
                face_resized = cv2.resize(face_crop, (FACE_INPUT_SIZE, FACE_INPUT_SIZE))

                results.append({
                    "face": face_resized,
                    "box": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "landmarks": {
                        "left_eye": (float(lms[0][0]), float(lms[0][1])),
                        "right_eye": (float(lms[1][0]), float(lms[1][1])),
                        "nose": (float(lms[2][0]), float(lms[2][1])),
                        "mouth_left": (float(lms[3][0]), float(lms[3][1])),
                        "mouth_right": (float(lms[4][0]), float(lms[4][1])),
                    },
                })

        return results

    def extract_single_face(self, frame: np.ndarray):
        """Extract the most confident face from frame. Returns None if no face found."""
        faces = self.extract_faces(frame)
        if not faces:
            return None
        # Return highest confidence face
        return max(faces, key=lambda f: f["confidence"])
