import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tempfile
import os

# MediaPipe 初始化
mp_pose = mp.solutions.pose

# 模型路徑
MODEL_PATH = '/Users/norvol/Desktop/專題/MediaPipe_Git/Project_mediaPipe/Data/random_forest_model.joblib'

# 載入模型
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model = load_model(MODEL_PATH)

def get_pose_class(prediction):
    """獲取姿勢類別"""
    pose_classes = ['Boxing', 'Normal', 'Squat', 'Raise']
    return pose_classes[prediction]

def process_video(video_path):
    """處理視頻並識別姿勢"""
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
                
                if pose_class != current_segment['pose']:
                    if current_segment['pose'] in ['Raise', 'Boxing', 'Squat']:
                        duration = (frame_count - current_segment['start']) / fps
                        if duration >= 2:  # 只保存持續時間超過2秒的片段
                            segments[current_segment['pose']].append((current_segment['start'], frame_count))
                    current_segment = {'pose': pose_class, 'start': frame_count}
            
            frame_count += 1
    
    cap.release()
    return segments, fps

def create_segment_video(video_path, segments, pose_type, fps):
    """創建指定姿勢的視頻片段"""
    cap = cv2.VideoCapture(video_path)
    
    if not segments[pose_type]:
        cap.release()
        return None
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file_path = temp_file.name
    temp_file.close()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    """主函數"""
    st.title("動作分類視頻分段器")

    # 登出按鈕
    if st.sidebar.button("登出"):
        st.session_state['logged_in'] = False
        st.rerun()

    uploaded_file = st.file_uploader("選擇視頻文件", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        
        if st.button("處理視頻"):
            with st.spinner('正在處理視頻...'):
                segments, fps = process_video(tfile.name)
                st.session_state['segments'] = segments
                st.session_state['video_path'] = tfile.name
                st.session_state['fps'] = fps
            st.success('視頻處理完成！')
        
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
                        st.write("沒有檢測到持續2秒以上的 '拿獎品' 動作")
            
            with col2:
                if st.button('攻擊'):
                    segment_video = create_segment_video(st.session_state['video_path'], 
                                                         st.session_state['segments'], 
                                                         'Boxing', 
                                                         st.session_state['fps'])
                    if segment_video:
                        st.video(segment_video)
                    else:
                        st.write("沒有檢測到持續2秒以上的 '攻擊' 動作")
            
            with col3:
                if st.button('取物'):
                    segment_video = create_segment_video(st.session_state['video_path'], 
                                                         st.session_state['segments'], 
                                                         'Squat', 
                                                         st.session_state['fps'])
                    if segment_video:
                        st.video(segment_video)
                    else:
                        st.write("沒有檢測到持續2秒以上的 '取物' 動作")

if __name__ == "__main__":
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.warning("請先登錄")
        st.stop()
    main()