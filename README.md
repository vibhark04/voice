# Speech Detection System for Verbally Impaired People

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red)](https://pytorch.org/)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)  
[![Made with ❤️](https://img.shields.io/badge/Made%20with-Deep%20Learning-red)](#)

---

## 📖 Overview  

This project explores how **muscle signals (EMG)** recorded from the face/jaw can be translated into **natural-sounding speech** using **deep learning**.  

The pipeline combines:  
- **Temporal Convolutional Networks (TCN)** for mapping EMG → MFCCs (speech features).  
- **Generative Adversarial Networks (GAN)** for generating realistic audio.  

The ultimate goal is a **silent speech interface** — technology that allows communication without vocalizing aloud.  

---

## 🏗️ System Architecture  

flowchart TD
    A [Silent EMG Input] --> B[Translator (TCN)]
    B --> C[Speech Features (MFCCs)]
    C --> D[Artist (Generator)]
    D --> E[Judge (Discriminator)]
    E -->|feedback| D
    D --> F[Generated Audio]
    F --> G[ASR Transcription & Evaluation]
📂 Data Requirements
Voiced EMG (*_emg.npy) → Muscle activity while speaking aloud

Voiced Audio (*_audio_clean.flac) → Ground truth speech

Silent EMG (*_emg.npy) → Muscle activity while mouthing silently

Helper Files (.json, _button.npy) → Timing and segmentation

⚙️ Installation
Clone the repo and install dependencies:

bash
Copy code
git clone https://github.com/your-username/silent-emg-speech.git
cd silent-emg-speech
pip install -r requirements.txt
Core Libraries:

PyTorch

NumPy

SciPy

Librosa

Matplotlib

Scikit-learn

▶️ Usage
1. Training
python
Copy code
from pipeline import train_pipeline

train_pipeline(
    silent_emg_file="3_emg.npy",
    voiced_emg_file="1_emg.npy",
    voiced_audio_file="1_audio_clean.flac",
    json_file="timing.json",
    silent_button_file="3_button.npy"
)
2. Inference / Testing
python
Copy code
from pipeline import test_pipeline

test_pipeline(
    new_emg_file="8_emg.npy",
    json_file="timing.json",
    button_file="8_button.npy"
)
📊 Evaluation Metrics
The system generates a report card including:

WER (Word Error Rate) → Speech accuracy

STOI (Short-Time Objective Intelligibility) → Speech clarity

MCD (Mel-Cepstral Distortion) → Voice similarity

Mel Spectrograms → Visual sound comparison

Precision / Recall / F1 + Confusion Matrix → Command classification

📌 Outputs
🎵 Synthesized Audio from silent EMG

📝 Text Transcription (ASR)

📈 Evaluation Report with metrics & plots

🔮 Future Work
Extend datasets for multilingual silent speech.

Explore transformer-based architectures.

Real-time EMG-to-speech applications for assistive devices.
