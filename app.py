# import streamlit as st
# import cv2 
# import numpy as np
# import mediapipe as mp
# from PIL import Image
# import pickle

# # Initialize Mediapipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

# # Load pretrained model and scaler
# loaded_model = pickle.load(open("finalmodel", "rb"))
# loaded_scaler = pickle.load(open("finalscaler", "rb"))

# # Function to extract landmarks from video frame
# def extract_landmarks(image):
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
#         row = []
#         for landmark in landmarks[:23]:
#             row.extend([landmark.x, landmark.y, landmark.z])
#         # Draw landmarks on the image frame
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         return row, image
#     else:
#         return None, image

# def main():
#     st.title("Pose Detection Using Image")
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Convert the file to an OpenCV image.
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)

#         # Extract landmarks
#         landmarks, landmarked_image = extract_landmarks(image)

#         if landmarks:
#             features_scaled = loaded_scaler.transform([landmarks])
#             prediction = loaded_model.predict(features_scaled)
#             label = f'Predicted Category: {prediction[0]}'
            
#             # Resizing the landmarked image to a fixed size
#             fixed_size = (400, 500)  
#             landmarked_image_resized = cv2.resize(landmarked_image, fixed_size, interpolation=cv2.INTER_AREA)
            
#             # Display the image with landmarks and label
#             st.image(cv2.cvtColor(landmarked_image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)
#             st.markdown(f"<h4 style='text-align: center; color: blue;'>{label}</h4>", unsafe_allow_html=True)
#         else:
#             st.write("No pose landmarks detected in the image.")

#     st.title("Real-time Pose Estimation")
#     st.info("Go to the Sidebar and select 'Use Webcam' to start real-time pose estimation.")

#     # Use Webcam
#     use_webcam = st.sidebar.button('Use Webcam', key='start_webcam')
#     stop_webcam = st.sidebar.button('Stop Webcam', key='stop_webcam')

#     if use_webcam:
#         confidence_threshold = 0.7
#         cap = cv2.VideoCapture(0)
#         stframe = st.empty()  # Placeholder for the video frames

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 st.write("Unable to access the webcam.")
#                 break

#             # Extract landmarks from the current frame
#             landmarks, landmarked_image = extract_landmarks(frame)

#             if landmarks:  # Check if landmarks were detected
#                 if len(landmarks) == loaded_model.n_features_in_:  # Ensure feature length matches model input
#                     features_scaled = loaded_scaler.transform([landmarks])
                    
#                     # Get the prediction and confidence scores
#                     prediction = loaded_model.predict(features_scaled)
#                     if hasattr(loaded_model, "predict_proba"):
#                         confidence = loaded_model.predict_proba(features_scaled).max()  # Use predict_proba for classifiers
#                     elif hasattr(loaded_model, "decision_function"):
#                         confidence = loaded_model.decision_function(features_scaled).max()  # Use decision_function for SVM
#                     else:
#                         confidence = None  # Fallback if the model does not provide confidence scores
                    
#                     # Display the predicted category and confidence score on the image
#                     if confidence is None or confidence >= confidence_threshold:
#                         text = f'Category: {prediction[0]}, Confidence: {confidence:.2f}' if confidence else f'Category: {prediction[0]}'
#                         cv2.putText(landmarked_image, text, 
#                                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
#             # Convert the frame to RGB for Streamlit display
#             stframe.image(cv2.cvtColor(landmarked_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
#             # Stop the webcam stream if 'Stop Webcam' is pressed
#             if stop_webcam:
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



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
    st.title("Pose Detection Using Image")
    
    # Image upload and processing
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Extract landmarks
        landmarks, landmarked_image = extract_landmarks(image, static_image_mode=True)

        if landmarks:
            features_scaled = loaded_scaler.transform([landmarks])
            prediction = loaded_model.predict(features_scaled)
            confidence = None
            if hasattr(loaded_model, "predict_proba"):
                confidence = loaded_model.predict_proba(features_scaled).max()  # Use predict_proba for classifiers
            elif hasattr(loaded_model, "decision_function"):
                confidence = loaded_model.decision_function(features_scaled).max()  # Use decision_function for SVM
            
            # Display the image with landmarks and label if confidence is sufficient
            if confidence is None or confidence >= 0.1:
                label = f'Predicted Category: {prediction[0]}, Confidence: {confidence:.2f}' if confidence else f'Predicted Category: {prediction[0]}'
                # Resizing the landmarked image to a fixed size
                fixed_size = (400, 400)  
                landmarked_image_resized = cv2.resize(landmarked_image, fixed_size, interpolation=cv2.INTER_AREA)
                
                # Display the image with landmarks and label
                st.image(cv2.cvtColor(landmarked_image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.markdown(f"<h4 style='text-align: center; color: blue;'>{label}</h4>", unsafe_allow_html=True)
            else:
                st.write(f"Confidence below threshold. No pose detected.{confidence}")
        else:
            st.write("No pose landmarks detected in the image.")

    st.title("Real-time Pose Estimation")
    st.info("Go to the Sidebar and select 'Use Webcam' to start real-time pose estimation.")

    # Use Webcam
    use_webcam = st.sidebar.button('Use Webcam', key='start_webcam')
    stop_webcam = st.sidebar.button('Stop Webcam', key='stop_webcam')

    if use_webcam:
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
                    if confidence is None or confidence >= 0.7:
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
