from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import mediapipe as mp
import numpy as np
from fastapi import WebSocket

app = FastAPI()

class FacialRecognitionSystem:
    def __init__(self, reference_image_path):
        """Initialize the facial recognition system"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load the reference image and compute features
        print(f"Loading reference image from: {reference_image_path}")
        image = cv2.imread(reference_image_path)
        if image is None:
            raise ValueError(f"Could not load image at {reference_image_path}")
        
        # Convert and get features
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in reference image")
            
        self.reference_features = self.extract_features(results.multi_face_landmarks[0])
        print("Reference features extracted successfully")
    
    def extract_features(self, landmarks):
        """Extract facial features from landmarks"""
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)
    
    def compare_features(self, features1, features2):
        """Compare two sets of facial features"""
        if features1 is None or features2 is None:
            return False, 0.0
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(features1 - features2)
        similarity = 1 / (1 + distance)  # Convert to similarity score (inverse of distance)
        
        return similarity
    
    def generate_video_feed(self):
        """Generate video feed and process frames"""
        video_capture = cv2.VideoCapture(0)  # Start the webcam capture
        if not video_capture.isOpened():
            raise RuntimeError("Could not open video capture")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            confidence = 0.0  # Default confidence
            if results.multi_face_landmarks:
                current_features = self.extract_features(results.multi_face_landmarks[0])
                confidence = self.compare_features(self.reference_features, current_features)
                is_match = confidence > 0.85
                color = (0, 255, 0) if is_match else (0, 0, 255)
                status = "Authorized: Fred Durst Confirmed" if is_match else "You are NOT Fred Durst!"

                for landmark in results.multi_face_landmarks[0].landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Draw the status and confidence score on the frame
                cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame with the status and confidence score overlay
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Initialize the facial recognition system
system = FacialRecognitionSystem("fred.jpg")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(system.generate_video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")
