import numpy as np
import librosa
import pickle
from rich.console import Console

"""
Predict the spoken word from the audio clip
Note: The audio clip should be in .wav format and should be recorded at 16kHz
      audio clip should be of single word, from 1 to 10 in Kashmiri Language.
"""
audio_clip_path = "/audio.wav"


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=25)
    return np.mean(mfccs.T, axis=0)


"""
Load the trained model, model.pkl
"""
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

"""
Feature extraction and prediction
"""
recorded_features = extract_features(audio_clip_path)
recorded_features = recorded_features.reshape(1, -1)
prediction = model.predict(recorded_features)
label_mapping = {
    0: "ten",
    1: "six",
    2: "eight",
    3: "one",
    4: "five",
    5: "two",
    6: "nine",
    7: "seven",
    8: "four",
    9: "three",
}
predicted_word = label_mapping[prediction[0]]

"""
Print the predicted word
"""
console = Console()
console.print("[bold blue]Predicted word:[/bold blue]")
console.print(f"[bold green on white]{predicted_word}[/bold green on white]")
