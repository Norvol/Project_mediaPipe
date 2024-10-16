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

# 初始化 MediaPipe，包含畫圖工具和姿勢模型
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 可用的模型選項及其路徑
MODEL_CHOICES = {
    'rf': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/random_forest_model.joblib',
    'svm': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/support_vector_machine_model.joblib',
    'nn': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/neural_network_model.joblib'
}

# FTP 配置，用於上傳警報文件
FTP_SERVER = '140.128.102.142'
FTP_PORT = 21
FTP_USERNAME = 'isslabView'
FTP_PASSWORD = 'isslab411view114'
FTP_ALARM_PATH = 'C:/Users/ISSLAB/Downloads/alarm'  # FTP 服務器上的路徑

# 警報條件設置
ALARM_COOLDOWN = 30  # 警報之間的冷卻時間（秒）
BOXING_DURATION_THRESHOLD = 2  # "Boxing" 姿勢持續時間的閾值（秒）

# 配置日誌，用於記錄應用的運行狀況
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_model(model_key):
    """
    加載選定的姿勢分類模型。

    :param model_key: 模型的鍵（如 'rf', 'svm', 'nn'）
    :return: 加載好的模型
    """
    if model_key not in MODEL_CHOICES:
        raise ValueError(f"Invalid model key. Choose from {', '.join(MODEL_CHOICES.keys())}")
    model_path = MODEL_CHOICES[model_key]
    return joblib.load(model_path)

def get_pose_class(prediction):
    """
    根據模型預測的結果返回對應的姿勢類別。

    :param prediction: 模型預測的數字結果
    :return: 姿勢類別名稱
    """
    pose_classes = ['Boxing', 'Normal', 'Squat', 'Raise']
    return pose_classes[prediction]

def send_alarm_to_ftp(timestamp):
    """
    生成警報文件並上傳到 FTP 服務器。

    :param timestamp: 當前的時間戳，用於生成警報文件名稱
    """
    alarm_file_name = f'boxing_alarm_{timestamp}.txt'
    alarm_content = f"Boxing detected at {timestamp}"

    try:
        # 創建臨時文件儲存警報內容
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(alarm_content)
            temp_file_path = temp_file.name

        # 連接到 FTP 服務器並上傳文件
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
        # 刪除臨時文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def main():
    """
    Streamlit 應用的主函數，用於姿勢分類和警報檢測。
    """
    st.title("Pose Classification App")

    # 在側邊欄中讓用戶選擇姿勢分類模型
    selected_model = st.selectbox("Select Model", list(MODEL_CHOICES.keys()))
    model = load_model(selected_model)

    # 用戶選擇視頻源：網絡攝像頭或上傳的視頻文件
    source_option = st.radio("Select video source", ('Webcam', 'Upload Video'))
    
    # 如果選擇上傳視頻文件
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
        # 選擇網絡攝像頭作為視頻源
        vf = cv2.VideoCapture(0)

    # 在頁面上創建一個空框，用於顯示處理後的視頻
    stframe = st.empty()
    
    # 初始化一個固定大小的隊列，用於存儲最近的預測結果
    recent_predictions = deque(maxlen=10)
    last_alarm_time = 0  # 記錄上次發送警報的時間
    boxing_start_time = None  # 記錄開始"Boxing"的時間
    boxing_accumulated_time = 0  # 累積"Boxing"的持續時間

    # 初始化 MediaPipe 的姿勢檢測模型
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while vf.isOpened():
            ret, img = vf.read()
            if not ret:
                st.warning("Can't receive frame (stream end?). Exiting ...")
                break
            
            # 調整視頻大小並進行顏色轉換
            img = cv2.resize(img, (640, 480))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            # 如果檢測到姿勢
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # 將姿勢數據轉換為平面數組
                pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
                prediction = model.predict([pose_data])[0]  # 使用模型進行預測
                recent_predictions.append(prediction)
                
                # 計算最近幾次預測中的最常見結果
                most_common = max(set(recent_predictions), key=recent_predictions.count)
                pose_class = get_pose_class(most_common)
                confidence = recent_predictions.count(most_common) / len(recent_predictions)
                
                # 繪製姿勢數據在視頻畫面上
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # 顯示當前的姿勢分類和信心度
                cv2.putText(img, f"Pose: {pose_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 記錄當前時間
                current_time = time.time()
                # 如果檢測到 "Boxing" 並且信心度高於 0.8
                if pose_class == 'Boxing' and confidence > 0.8:
                    if boxing_start_time is None:
                        boxing_start_time = current_time  # 記錄"Boxing"的開始時間
                    else:
                        # 累加"Boxing"的持續時間
                        boxing_accumulated_time += current_time - boxing_start_time
                        boxing_start_time = current_time
                else:
                    # 重置"Boxing"的開始時間
                    boxing_start_time = None
                
                # 如果累積的"Boxing"時間超過閾值且冷卻時間已過，則發送警報
                if boxing_accumulated_time >= BOXING_DURATION_THRESHOLD and (current_time - last_alarm_time) > ALARM_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    send_alarm_to_ftp(timestamp)
                    last_alarm_time = current_time  # 更新上次警報時間
                    boxing_accumulated_time = 0  # 重置累積時間
            
            # 在 Streamlit 界面上顯示處理後的視頻畫面
            stframe.image(img, channels="BGR")
            
            # 按 'Esc' 退出循環
            if cv2.waitKey(5) & 0xFF == 27:
                break

    vf.release()

if __name__ == "__main__":
    main()

    # 使用 Streamlit 運行此應用程序的命令
    # streamlit run poseTest.py
