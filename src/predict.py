import json
import joblib
import numpy as np
import librosa
from pathlib import Path
import serial
import time
import sounddevice as sd
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "animal_sound_rf.joblib"
CONFIG_PATH = PROJECT_DIR / "models" / "model_config.json"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CONFIDENCE_THRESHOLD = config["confidence_threshold"]

# Arduino bağlantısı
ARDUINO_PORT = "COM3"

arduino = serial.Serial(
    ARDUINO_PORT,
    9600,
    timeout=1
)

# Arduino reset atınca hazır olması için
time.sleep(2)

print("Arduino connected:", ARDUINO_PORT)

print("Model loaded successfully.")
print("Classes:", model.classes_)
print("Threshold:", CONFIDENCE_THRESHOLD)
def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=config["sample_rate"],
        duration=config["duration_seconds"]
    )

    # Sessiz / çok düşük sesli kısımları kaldır
    y_trimmed, _ = librosa.effects.trim(
        y,
        top_db=30
    )

    # Eğer trim sonucunda neredeyse hiç ses kalmadıysa
    # orijinal kaydı kullan
    if len(y_trimmed) > 0:
        y = y_trimmed

    target_length = (
        config["sample_rate"]
        * config["duration_seconds"]
    )

    # 5 saniyeden kısaysa tekrar ederek tamamla
    if len(y) < target_length:
        repeats = int(np.ceil(target_length / len(y)))
        y = np.tile(y, repeats)[:target_length]
    else:
        y = y[:target_length]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Delta MFCC
    delta = librosa.feature.delta(mfcc)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(y)

    spectral_features = [
        np.mean(centroid),
        np.std(centroid),

        np.mean(bandwidth),
        np.std(bandwidth),

        np.mean(rolloff),
        np.std(rolloff),

        np.mean(zcr),
        np.std(zcr)
    ]

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std,
        chroma_mean,
        chroma_std,
        spectral_features
    ])

    return features

def record_microphone(output_path):

    sr = config["sample_rate"]
    duration = config["duration_seconds"]

    print(f"\n🎤 Listening for {duration} seconds...")

    recording = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype="float32",
        device=1
    )

    sd.wait()

    # Mono array
    recording = recording.flatten()

    # Kayıt ses seviyesini göster
    peak = np.max(np.abs(recording))
    rms = np.sqrt(np.mean(recording ** 2))

    print(f"Peak level: {peak:.4f}")
    print(f"RMS level:  {rms:.4f}")

    sf.write(output_path, recording, sr)

    print("Recording finished.")


def predict_audio(audio_path):

    features = extract_features(audio_path)

    # Model tek örneği (1, 84) şeklinde bekliyor
    features = features.reshape(1, -1)

    # Her sınıf için probability
    probabilities = model.predict_proba(features)[0]

    # En yüksek probability'nin konumu
    best_index = np.argmax(probabilities)

    predicted_class = model.classes_[best_index]
    confidence = probabilities[best_index]

    # Confidence yeterli mi?
    if confidence >= CONFIDENCE_THRESHOLD:
        result = predicted_class
        accepted = True
    else:
        result = "UNCERTAIN"
        accepted = False

    return {
        "result": result,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "accepted": accepted
    }

if __name__ == "__main__":

    live_audio_path = PROJECT_DIR / "live_audio.wav"

    while True:

        input("\nPress ENTER to listen...")

        # 5 saniye mikrofonu dinle
        record_microphone(live_audio_path)

        # Model tahmini
        prediction = predict_audio(live_audio_path)

        print("\n--- Prediction ---")
        print("Predicted class:", prediction["predicted_class"])
        print(
            "Confidence:",
            round(prediction["confidence"] * 100, 2),
            "%"
        )
        print("Final result:", prediction["result"])

        # Arduino'ya sonucu gönder
        message = prediction["result"]
        arduino.write((message + "\n").encode())

        print("Sent to Arduino:", message)