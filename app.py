import streamlit as st
import cv2 
import numpy as np
import mediapipe as mp
import pickle

# Initialize MediaPipe Pose once
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# Load pretrained model and scaler
loaded_model = pickle.load(open("finalmodel", "rb"))
loaded_scaler = pickle.load(open("finalscaler", "rb"))

# Function to extract landmarks from an image or video frame
def extract_landmarks(image, static_image_mode=False):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = []
        for landmark in landmarks[:23]:  # Ensure only 23 landmarks are considered
            row.extend([landmark.x, landmark.y, landmark.z])
        # Draw landmarks on the image frame
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return row, image
    else:
        return None, image

def main():
    st.title("Real-time Pose Detection")

    # Use Webcam
    start_webcam = st.button('Start Real-time Pose Detection', key='start_webcam')
    stop_webcam = st.button('Stop Real-time Pose Detection', key='stop_webcam')

    if start_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()  # Placeholder for the video frames

        # Loop to capture video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to access the webcam.")
                break

            # Extract landmarks from the current frame
            landmarks, landmarked_image = extract_landmarks(frame)

            if landmarks:
                if len(landmarks) == loaded_model.n_features_in_:  # Ensure feature length matches model input
                    features_scaled = loaded_scaler.transform([landmarks])
                    
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
                        cv2.putText(landmarked_image, text, 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Convert the frame to RGB for Streamlit display
            stframe.image(cv2.cvtColor(landmarked_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Stop the webcam stream if 'Stop Webcam' is pressed
            if stop_webcam:
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
