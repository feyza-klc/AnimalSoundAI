#Animal Sound AI

A real-time animal sound classification and confidence-aware alert system built with Python, Machine Learning, and Arduino.

The system classifies five animal sounds — **cat, dog, cow, sheep, and frog** — and uses prediction confidence to decide whether a result should be accepted or marked as **UNCERTAIN**.

The project also investigates how environmental noise affects audio classification and how noise augmentation can improve model robustness.

---

##  Project Overview

Animal Sound AI combines a machine learning pipeline with a physical Arduino interface.

The complete system works as follows:

**Microphone → Audio Processing → ML Model → Confidence Check → Serial Communication → Arduino → OLED + LEDs**

The computer records audio and extracts acoustic features. A Random Forest model predicts the animal class and produces a confidence score.

The prediction is then evaluated using a confidence threshold:

- 🟢 **Green LED:** prediction accepted
- 🔴 **Red LED:** uncertain prediction
- 📟 **OLED:** displays the prediction status

---

## Classes

The model recognizes five animal sound classes:

- Cat
- Dog
- Cow
- Sheep
- Frog

The dataset was created from the **ESC-50 Environmental Sound Classification Dataset**, using 40 recordings from each class for a total of **200 audio samples**.

---

##  Feature Extraction

Each audio recording is represented using **84 acoustic features** extracted with Librosa:

- MFCC mean and standard deviation
- Delta MFCC mean and standard deviation
- Chroma features
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero-crossing rate

---

## Model Comparison

Several approaches were evaluated using the official ESC-50 folds.

| Model | Mean Accuracy |
|---|---:|
| Random Forest – Baseline Features | 65.0% |
| SVM – Enhanced Features | 71.5% |
| **Random Forest – Enhanced Features** | **74.0%** |

A CNN was also investigated using Mel spectrograms, but its cross-validation performance was less stable on the small 200-sample dataset.

The enhanced Random Forest was therefore selected as the final model.

---

##  Noise Robustness

To evaluate robustness, artificial Gaussian noise was added at different Signal-to-Noise Ratio (SNR) levels.

### Model trained on clean audio

| Test Condition | Accuracy |
|---|---:|
| Clean | 74.0% |
| 20 dB SNR | 57.0% |
| 10 dB SNR | 30.0% |
| 0 dB SNR | 21.5% |

The results showed a significant performance decrease under noisy conditions.

### Noise-Augmented Training

The training data was augmented using **20 dB and 10 dB SNR noise**.

| Test Condition | Accuracy |
|---|---:|
| Clean | 65.0% |
| 20 dB SNR | 79.0% |
| 10 dB SNR | 78.5% |
| 0 dB SNR | 51.5% |

Noise augmentation substantially improved robustness, including at the unseen 0 dB noise level, although clean-audio accuracy decreased.

---


## Confidence-Aware Prediction

Instead of accepting every prediction, the system includes a confidence-based rejection mechanism.

During controlled evaluation, a **0.50 confidence threshold** provided:

- **44.1% prediction coverage**
- **90.1% accuracy among accepted predictions**

This demonstrates the trade-off between prediction coverage and reliability.

During live microphone testing, confidence scores were generally lower due to the domain shift between ESC-50 recordings and real-world microphone input. Therefore, the real-time prototype uses a configurable **0.35 confidence threshold**.

The threshold is stored in `model_config.json` and can be adjusted depending on the deployment environment.

Predictions below the selected threshold are returned as:

```text
UNCERTAIN

---

## Real-Time Prototype

The final prototype captures audio from a computer microphone and performs inference in Python.

The result is sent to an Arduino through serial communication.

Example:

```text
Listening for 5 seconds...

Predicted class: dog
Confidence: 43.0%
Final result: dog

Sent to Arduino: dog
```

The Arduino then updates the OLED display and LEDs according to the received result.

---

##  Hardware

The prototype uses:

- Arduino
- I2C OLED display
- Green LED
- Red LED
- 220–330 Ω resistors
- Breadboard
- Jumper wires

Machine learning inference runs on the computer. The Arduino acts as the physical output and alert interface.

---

##  Real-World Limitations

Although the model performs well under controlled ESC-50 evaluation, microphone recordings differ from the training dataset.

Live testing showed lower confidence values even when the predicted class was correct. This indicates a **dataset-to-real-world domain shift**.

Possible future improvements include:

- Collecting real microphone recordings
- Training with microphone-specific augmentation
- Improving audio segmentation
- Expanding the number of animal classes
- Testing on a larger external dataset
- Deploying a lightweight model directly on an embedded device

---

##  Technologies

**Machine Learning**
- Python
- Scikit-learn
- TensorFlow / Keras
- Librosa
- NumPy

**Embedded System**
- Arduino
- OLED Display
- Serial Communication

**Audio**
- SoundDevice
- SoundFile

---

## 📁 Project Structure

```text
AnimalSoundAI/
│
├── ESC-50/
├── models/
│   ├── animal_sound_rf.joblib
│   └── model_config.json
│
├── notebooks/
│   ├── 01_classical_ml.ipynb
│   ├── 02_cnn.ipynb
│   └── 03_noise_robustness.ipynb
│
├── src/
│   └── predict.py
│
├── test_audio/
├── README.md
└── requirements.txt
```

---

##  Key Takeaways

This project explores more than classification accuracy alone.

It demonstrates:

- Audio feature engineering
- Classical ML and deep learning comparison
- Cross-validation
- Noise robustness testing
- Data augmentation
- Confidence-aware rejection
- Real-time microphone inference
- Python–Arduino serial communication
- Physical ML system prototyping

The project highlights an important real-world ML challenge: **a model that performs well on a controlled dataset may behave differently when deployed in a real environment.**

---

##  Author

Developed as a machine learning and embedded systems project.
