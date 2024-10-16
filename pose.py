#ftp版本----------
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import os
import time
from ftplib import FTP
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe 初始化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 模型選擇
MODEL_CHOICES = {
    'rf': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/random_forest_model.joblib',
    'svm': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/support_vector_machine_model.joblib',
    'nn': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/neural_network_model.joblib'
}
SELECTED_MODEL = 'rf'

# FTP 配置
FTP_SERVER = '140.128.102.142'  #'140.128.102.142'
FTP_PORT = 21
FTP_USERNAME = 'isslabView'
FTP_PASSWORD = 'isslab411view114'
FTP_ALARM_PATH = 'C:/Users/ISSLAB/Downloads/alarm'  # FTP 服務器上的路徑

# Alarm 條件
ALARM_COOLDOWN = 30  # 警報冷卻時間（秒）
BOXING_DURATION_THRESHOLD = 2  # 需要的 'Boxing' 次數

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
        # 在本地創建一個警報文件
        with open(alarm_file_name, 'w') as f:
            f.write(alarm_content)

        # 使用 FTP 傳送文件到服務器
        with FTP() as ftp:
            ftp.connect(FTP_SERVER, FTP_PORT)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            ftp.cwd(FTP_ALARM_PATH)
            with open(alarm_file_name, 'rb') as file:
                ftp.storbinary(f'STOR {alarm_file_name}', file)

        logging.info(f"Alarm sent to FTP: {FTP_ALARM_PATH}/{alarm_file_name}")

    except Exception as e:
        print()
        logging.error(f"Failed to send alarm to FTP: {str(e)}")
        print()
    # finally:
    #     # 刪除本地創建的警報文件
    #     if os.path.exists(alarm_file_name):
    #         os.remove(alarm_file_name)

def main():
    model = load_model(SELECTED_MODEL)
    recent_predictions = deque(maxlen=10)
    last_alarm_time = 0
    
    boxing_start_time = None
    boxing_accumulated_time = 0
    
    print("\nEnter 'c' for webcam or provide a video file path:")       # C:\Users\ISSLAB\Downloads\MediaPipe_MainOut\testv.mp4
    input_source = input().strip()
    
    cap = cv2.VideoCapture(0) if input_source.lower() == 'c' else cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                logging.info("Can't receive frame (stream end?). Exiting ...")
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
                # 如果檢測到 "Boxing" 且信心度大於 0.8
                if pose_class == 'Boxing' and confidence > 0.8:
                    if boxing_start_time is None:
                        # 開始新的 "Boxing" 計時
                        boxing_start_time = current_time
                    else:
                        # 累加 "Boxing" 的持續時間
                        boxing_accumulated_time += current_time - boxing_start_time
                        boxing_start_time = current_time

                else:
                    # 如果不是 "Boxing"，重置起始時間
                    boxing_start_time = None
                
                # 如果累積時間超過閾值並且冷卻時間已過，發送警報
                if boxing_accumulated_time >= BOXING_DURATION_THRESHOLD and (current_time - last_alarm_time) > ALARM_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    send_alarm_to_ftp(timestamp)
                    last_alarm_time = current_time
                    boxing_accumulated_time = 0  # 重置累積時間以防止重複觸發
            
            cv2.imshow('Pose Classification', img)
            if cv2.waitKey(5) == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()