import streamlit as st
import av
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pickle

# Initialize MediaPipe Pose once
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Load pretrained model and scaler
loaded_model = pickle.load(open("finalmodel", "rb"))
loaded_scaler = pickle.load(open("finalscaler", "rb"))

class PoseDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Extract landmarks from the current frame
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = []
            for landmark in landmarks[:23]:  # Ensure only 23 landmarks are considered
                row.extend([landmark.x, landmark.y, landmark.z])
            
            # Draw landmarks on the image frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Check if the landmarks match the expected number of features for the model
            if len(row) == loaded_model.n_features_in_:
                features_scaled = loaded_scaler.transform([row])
                
                # Get the prediction and confidence scores
                prediction = loaded_model.predict(features_scaled)
                confidence = None
                if hasattr(loaded_model, "predict_proba"):
                    confidence = loaded_model.predict_proba(features_scaled).max()  # Use predict_proba for classifiers
                elif hasattr(loaded_model, "decision_function"):
                    confidence = loaded_model.decision_function(features_scaled).max()  # Use decision_function for SVM
                
                # Display the predicted category and confidence score on the image if confidence is sufficient
                if confidence is None or confidence >= 0.6:
                    text = f'Category: {prediction[0]}, Confidence: {confidence:.2f}' if confidence else f'Category: {prediction[0]}'
                    cv2.putText(image, text, 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return image

def main():
    st.title("Real-time Pose Detection with Streamlit-WebRTC")

    webrtc_streamer(key="pose-detection", video_transformer_factory=PoseDetector)

if __name__ == "__main__":
    main()
