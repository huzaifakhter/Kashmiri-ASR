import os
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

"""
Extract features from the audio file using MFCC.
You can change the number of MFCC features extracted by changing the n_mfcc parameter.
"""

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=25)
    return np.mean(mfccs.T, axis=0)


"""
Load the dataset and extract features.
Change the dataset_path to the path of the dataset containing the audio files.

"""
dataset_path = "/AUDIO_PATH"

labels = {folder: i for i, folder in enumerate(os.listdir(dataset_path))}
print("Labels Mapping:", labels)

X, y = [], []

for word, label in labels.items():
    word_path = os.path.join(dataset_path, word)
    if os.path.isdir(word_path):
        for file in os.listdir(word_path):
            if file.endswith(".wav"):
                file_path = os.path.join(word_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sample X:", X[:2])
print("Sample y:", y[:2])

"""
Train the model using RandomForestClassifier
"""
model = RandomForestClassifier(n_estimators=100, random_state=42)
stratified_kf = StratifiedKFold(n_splits=3)
cv_scores = cross_val_score(model, X, y, cv=stratified_kf)

"""
Print the cross-validation scores and the average accuracy
"""
print(f"Cross-validation scores: {cv_scores}")
print(f"Average accuracy: {cv_scores.mean() * 100:.2f}%")
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

"""
Print the model accuracy on the full dataset
"""
print(f"Model accuracy on the full dataset: {accuracy * 100:.2f}%")
print(f"Predictions: {y_pred}")
print(f"Actual labels: {y}")


"""
Save the trained model to a pickle file.
"""
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
