import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Model selection
MODEL_CHOICES = {
    'rf': 'C:\Users\ISSLAB\Downloads\MediaPipe_MainOut\Data\random_forest_model.joblib',
    'svm': 'C:\Users\ISSLAB\Downloads\MediaPipe_MainOut\Data\support_vector_machine_model.joblib',
    'nn': 'C:\Users\ISSLAB\Downloads\MediaPipe_MainOut\Data\neural_network_model.joblib'
}

# Choose your model here
SELECTED_MODEL = 'rf'  # Options: 'rf', 'svm', 'nn'

def load_model(model_key):
    if model_key not in MODEL_CHOICES:
        raise ValueError(f"Invalid model key. Choose from {', '.join(MODEL_CHOICES.keys())}")
    model_path = MODEL_CHOICES[model_key]
    return joblib.load(model_path)

def get_pose_class(prediction):
    pose_classes = ['Boxing', 'Normal', 'Squat', 'Raise']
    return pose_classes[prediction]

def main():
    model = load_model(SELECTED_MODEL)
    recent_predictions = deque(maxlen=10)
    
    # Ask user for input source
    print()
    input_source = input("Enter 'c' for webcam or provide a video file path: ")
    print()
    
    if input_source.lower() == 'c':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
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
            
            cv2.imshow('Pose Classification', img)
            if cv2.waitKey(5) == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
