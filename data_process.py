import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

def extract_landmarks(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            else:
                print(f"No pose landmarks detected in: {image_path}")
                return None
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Paths to your image folders
base_folder = '/Users/norvol/Desktop/專題/MediaPipe_Main/Pose_image'
pose_folders = {
    'Boxing': os.path.join(base_folder, 'Boxing'),
    'Normal': os.path.join(base_folder, 'Normal'),
    'Squat': os.path.join(base_folder, 'Squat'),
    'Raise': os.path.join(base_folder, 'Raise')
}

x = []  # Features
y = []  # Labels

# Extract landmarks for all pose types
for pose_label, folder_path in pose_folders.items():
    for image_file in os.listdir(folder_path):
        landmarks = extract_landmarks(os.path.join(folder_path, image_file))
        if landmarks is not None:
            x.append(landmarks)
            y.append(list(pose_folders.keys()).index(pose_label))

x = np.array(x)
y = np.array(y)

# Save the extracted features and labels
np.save('/Users/norvol/Desktop/專題/MediaPipe_Main/Data/pose_features.npy', x)
np.save('/Users/norvol/Desktop/專題/MediaPipe_Main/Data/pose_labels.npy', y)

print(f"Extracted features from {len(x)} images")
print(f"Label distribution: {np.bincount(y)}")