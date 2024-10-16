import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# Load the data
x = np.load('/Data/pose_features.npy')
y = np.load('/Data/pose_labels.npy')

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Function to evaluate and train model
def evaluate_and_train(model, name, save_path):
    # Cross-validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    print(f"\n{name} Cross-Validation Results:")
    print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on training set and evaluate on test set
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\n{name} Test Set Results:")
    print(classification_report(y_test, y_pred, target_names=['Boxing', 'Normal', 'Squat', 'Raise']))
    
    # Train final model on full dataset and save
    final_model = model.__class__(**model.get_params())
    final_model.fit(x, y)
    joblib.dump(final_model, save_path)
    print(f"Model saved to {save_path}")

# List of models to evaluate with their save paths
models = [
    (RandomForestClassifier(n_estimators=100, random_state=42),
     "Random Forest",
     '/Data/random_forest_model.joblib'),
    (SVC(kernel='rbf', random_state=42),
     "Support Vector Machine",
     '/Data/support_vector_machine_model.joblib'),
    (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
     "Neural Network",
     '/Data/neural_network_model.joblib')
]

# Evaluate and train each model
for model, name, save_path in models:
    evaluate_and_train(model, name, save_path)

print("\nAll models evaluated, trained on full dataset, and saved to specified paths.")