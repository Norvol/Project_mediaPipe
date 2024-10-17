import streamlit as st
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import os
import time
import tempfile
import logging
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 初始化 MediaPipe，包含畫圖工具和姿勢模型
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 可用的模型選項及其路徑
MODEL_CHOICES = {
    'rf': '/Users/norvol/Desktop/專題/MediaPipe_Git/Project_mediaPipe/Data/random_forest_model.joblib',
    'svm': '/Users/norvol/Desktop/專題/MediaPipe_Git/Project_mediaPipe/Data/support_vector_machine_model.joblib',
    'nn': '/Users/norvol/Desktop/專題/MediaPipe_Git/Project_mediaPipe/Data/neural_network_model.joblib'
}

# MODEL_CHOICES = {
#     'rf': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/random_forest_model.joblib',
#     'svm': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/support_vector_machine_model.joblib',
#     'nn': 'C:/Users/ISSLAB/Downloads/MediaPipe_MainOut/Data/neural_network_model.joblib'
# }



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

# 載入模型
@st.cache_resource
def load_model(model_key):
    if model_key not in MODEL_CHOICES:
        raise ValueError(f"Invalid model key. Choose from {', '.join(MODEL_CHOICES.keys())}")
    model_path = MODEL_CHOICES[model_key]
    return joblib.load(model_path)

model = load_model('rf')  # 請替換為您的模型路徑

def get_pose_class(prediction):
    """
    將模型的數字預測轉換為對應的姿勢類別名稱。
    """
    pose_classes = ['Boxing', 'Normal', 'Squat', 'Raise']
    return pose_classes[prediction]

def process_video(video_path):
    """
    處理影片文件，識別不同的姿勢段落。
    
    參數：
    video_path: 影片文件的路徑
    
    返回：
    segments: 包含每種姿勢的時間段的字典
    fps: 影片的幀率
    """
    segments = {'Raise': [], 'Boxing': [], 'Squat': []}
    current_segment = {'pose': None, 'start': 0}
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
                prediction = model.predict([pose_data])[0]
                pose_class = get_pose_class(prediction)
                
                # 如果姿勢改變，記錄新的段落
                if pose_class != current_segment['pose']:
                    if current_segment['pose'] in ['Raise', 'Boxing', 'Squat']:
                        segments[current_segment['pose']].append((current_segment['start'], frame_count))
                    current_segment = {'pose': pose_class, 'start': frame_count}
            
            frame_count += 1
    
    cap.release()
    return segments, fps

def create_segment_video(video_path, segments, pose_type, fps):
    """
    使用 OpenCV 根據指定的姿勢類型創建一個只包含該姿勢片段的新影片。
    
    參數：
    video_path: 原始影片的路徑
    segments: 包含各姿勢時間段的字典
    pose_type: 要提取的姿勢類型
    fps: 影片的幀率
    
    返回：
    生成的影片文件的路徑，如果沒有相應的片段則返回 None
    """
    cap = cv2.VideoCapture(video_path)
    
    if not segments[pose_type]:
        cap.release()
        return None
    
    # 創建臨時文件來保存輸出影片
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # 獲取原影片的寬度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 創建影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file_path, fourcc, fps, (width, height))
    
    for start, end in segments[pose_type]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    
    cap.release()
    out.release()
    
    return temp_file_path

def main():
    """
    主函數，設置 Streamlit 應用的用戶界面和邏輯流程。
    """
    st.title("動作分類影片分段器")

    # 上傳影片文件
    uploaded_file = st.file_uploader("選擇影片文件", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # 顯示上傳的影片
        st.video(tfile.name)
        
        # 處理影片按鈕
        if st.button("處理影片"):
            with st.spinner('正在處理影片...'):
                segments, fps = process_video(tfile.name)
                st.session_state['segments'] = segments
                st.session_state['video_path'] = tfile.name
                st.session_state['fps'] = fps
            st.success('影片處理完成！')
        
        # 如果影片已經處理，顯示動作按鈕
        if 'segments' in st.session_state:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button('拿獎品'):
                    segment_video = create_segment_video(st.session_state['video_path'], 
                                                         st.session_state['segments'], 
                                                         'Raise', 
                                                         st.session_state['fps'])
                    if segment_video:
                        st.video(segment_video)
                    else:
                        st.write("沒有檢測到 '拿獎品' 動作")
            
            with col2:
                if st.button('攻擊'):
                    segment_video = create_segment_video(st.session_state['video_path'], 
                                                         st.session_state['segments'], 
                                                         'Boxing', 
                                                         st.session_state['fps'])
                    if segment_video:
                        st.video(segment_video)
                    else:
                        st.write("沒有檢測到 '攻擊' 動作")
            
            with col3:
                if st.button('取物'):
                    segment_video = create_segment_video(st.session_state['video_path'], 
                                                         st.session_state['segments'], 
                                                         'Squat', 
                                                         st.session_state['fps'])
                    if segment_video:
                        st.video(segment_video)
                    else:
                        st.write("沒有檢測到 '取物' 動作")

if __name__ == "__main__":
    main()
    
    #streamlit run poseTest.py