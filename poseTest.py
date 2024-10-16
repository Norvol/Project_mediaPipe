import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import os
import time
from ftplib import FTP
import logging
import tempfile

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Model choices
MODEL_CHOICES = {
    'rf': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/random_forest_model.joblib',
    'svm': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/support_vector_machine_model.joblib',
    'nn': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/neural_network_model.joblib'
}

# FTP configuration
FTP_SERVER = '140.128.102.142'
FTP_PORT = 21
FTP_USERNAME = 'isslabView'
FTP_PASSWORD = 'isslab411view114'
FTP_ALARM_PATH = '/alarm'  # FTP server path

# Alarm conditions
ALARM_COOLDOWN = 30  # seconds
BOXING_DURATION_THRESHOLD = 2  # required 'Boxing' count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_model(model_key):
    if model_key not in MODEL_CHOICES:
        raise ValueError(f"Invalid model key. Choose from {', '.join(MODEL_CHOICES.keys())}")
    model_path = MODEL_CHOICES[model_key]
    return joblib.load(model_path)

def get_pose_class(prediction):
    pose_classes = ['Boxing', 'Normal', 'Squat', 'Raise']
    return pose_classes[prediction]

def send_alarm_to_ftp(timestamp):
    alarm_file_name = f'boxing_alarm_{timestamp}.txt'
    alarm_content = f"Boxing detected at {timestamp}"

    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(alarm_content)
            temp_file_path = temp_file.name

        with FTP() as ftp:
            ftp.connect(FTP_SERVER, FTP_PORT)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            ftp.cwd(FTP_ALARM_PATH)
            with open(temp_file_path, 'rb') as file:
                ftp.storbinary(f'STOR {alarm_file_name}', file)

        st.success(f"Alarm sent to FTP: {FTP_ALARM_PATH}/{alarm_file_name}")
        logging.info(f"Alarm sent to FTP: {FTP_ALARM_PATH}/{alarm_file_name}")

    except Exception as e:
        st.error(f"Failed to send alarm to FTP: {str(e)}")
        logging.error(f"Failed to send alarm to FTP: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def main():
    st.title("Pose Classification App")

    # Model selection
    selected_model = st.selectbox("Select Model", list(MODEL_CHOICES.keys()))
    model = load_model(selected_model)

    # Video source selection
    source_option = st.radio("Select video source", ('Webcam', 'Upload Video'))
    
    if source_option == 'Upload Video':
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a video file.")
            return
    else:
        vf = cv2.VideoCapture(0)

    stframe = st.empty()
    
    recent_predictions = deque(maxlen=10)
    last_alarm_time = 0
    boxing_start_time = None
    boxing_accumulated_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while vf.isOpened():
            ret, img = vf.read()
            if not ret:
                st.warning("Can't receive frame (stream end?). Exiting ...")
                break
            
            img = cv2.resize(img, (640, 480))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
                prediction = model.predict([pose_data])[0]
                recent_predictions.append(prediction)
                
                most_common = max(set(recent_predictions), key=recent_predictions.count)
                pose_class = get_pose_class(most_common)
                confidence = recent_predictions.count(most_common) / len(recent_predictions)
                
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                cv2.putText(img, f"Pose: {pose_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                current_time = time.time()
                if pose_class == 'Boxing' and confidence > 0.8:
                    if boxing_start_time is None:
                        boxing_start_time = current_time
                    else:
                        boxing_accumulated_time += current_time - boxing_start_time
                        boxing_start_time = current_time
                else:
                    boxing_start_time = None
                
                if boxing_accumulated_time >= BOXING_DURATION_THRESHOLD and (current_time - last_alarm_time) > ALARM_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    send_alarm_to_ftp(timestamp)
                    last_alarm_time = current_time
                    boxing_accumulated_time = 0
            
            stframe.image(img, channels="BGR")
            
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                break

    vf.release()

if __name__ == "__main__":
    main()

    # streamlit run poseTest.py    