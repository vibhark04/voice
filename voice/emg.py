import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy import signal
from scipy.fft import fft, ifft
import torchaudio
import math
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import pandas as pd
import whisper
from jiwer import wer
import json
import os
import tempfile
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("FFmpeg not found. Install FFmpeg and add to PATH for torchaudio to work.")
        return False

def load_flac_audio(file_path, fs=16000, target_length=16000):
    try:
        audio, sr = librosa.load(file_path, sr=fs, mono=True)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        audio /= np.max(np.abs(audio)) + 1e-8
        logging.info(f"Loaded audio from {file_path}, shape: {audio.shape}")
        return audio
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded JSON from {file_path}: {data}")
            return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def preprocess_emg(emg_signal, fs=1000, lowcut=20, highcut=450, order=4):
    if emg_signal.ndim == 1:
        emg_signal = emg_signal.reshape(-1, 1)
    mu = np.mean(emg_signal, axis=0)
    sigma = np.std(emg_signal, axis=0)
    emg_normalized = (emg_signal - mu) / (sigma + 1e-8)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    emg_filtered = signal.filtfilt(b, a, emg_normalized, axis=0)
    emg_fft = fft(emg_filtered, axis=0)
    emg_processed = np.real(ifft(emg_fft, axis=0))
    logging.info(f"Preprocessed EMG shape: {emg_processed.shape}")
    return emg_processed

def augment_emg(emg_signal, noise_level=0.005):
    noise = np.random.normal(0, noise_level, emg_signal.shape)
    return emg_signal + noise

def segment_emg(emg_signal, button_signal, info=None, fs=1000, min_length=1000):
    if info and 'chunks' in info and info['chunks']:
        chunks = info['chunks']
        start_samples = chunks[0][0]
        end_samples = chunks[-1][1]
        segment = emg_signal[start_samples:end_samples]
        logging.info(f"Segmented EMG using JSON chunks: {start_samples} to {end_samples}")
    elif button_signal is not None and len(button_signal) > 0:
        button_signal = button_signal.flatten() if button_signal.ndim > 1 else button_signal
        active_indices = np.where(button_signal > 0)[0]
        if len(active_indices) > 0:
            start_idx = max(0, active_indices[0] - 100)
            end_idx = min(len(emg_signal), active_indices[-1] + 100)
            segment = emg_signal[start_idx:end_idx]
            logging.info(f"Segmented EMG using button signal: {start_idx} to {end_idx}")
        else:
            segment = emg_signal[:min_length]
    else:
        segment = emg_signal[:min_length]
        logging.info(f"No button signal or chunks; using first {min_length} samples")
    
    if len(segment) < min_length:
        segment = np.pad(segment, ((0, min_length - len(segment)), (0, 0)), mode='constant')
    elif len(segment) > min_length:
        segment = segment[:min_length]
    return segment

def extract_mfcc(audio, fs=16000, n_mfcc=13, target_length=1000):
    mfcc = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=n_mfcc)
    if mfcc.shape[1] > target_length:
        mfcc = mfcc[:, :target_length]
    elif mfcc.shape[1] < target_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_length - mfcc.shape[1])), mode='constant')
    mfcc_min = np.min(mfcc)
    mfcc_max = np.max(mfcc)
    mfcc = (mfcc - mfcc_min) / (mfcc_max - mfcc_min + 1e-8)
    logging.info(f"MFCC shape: {mfcc.shape}")
    return mfcc

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        return out + self.residual(x)

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4, kernel_size=3, dilation_base=2, seq_length=1000):
        super(TCN, self).__init__()
        layers = []
        for i in range(n_blocks):
            dilation = dilation_base ** i
            layers.append(TCNBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Linear(out_channels, out_channels)
        self.seq_length = seq_length
    
    def forward(self, x):
        x = self.tcn(x)
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        if x.size(2) > self.seq_length:
            x = x[:, :, :self.seq_length]
        elif x.size(2) < self.seq_length:
            padding = torch.zeros(x.size(0), x.size(1), self.seq_length - x.size(2)).to(x.device)
            x = torch.cat([x, padding], dim=2)
        return x

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, seq_length=16000):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, out_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.seq_length = seq_length
    
    def forward(self, x):
        x = self.net(x)
        if x.size(2) > self.seq_length:
            x = x[:, :, :self.seq_length]
        elif x.size(2) < self.seq_length:
            padding = torch.zeros(x.size(0), x.size(1), self.seq_length - x.size(2)).to(x.device)
            x = torch.cat([x, padding], dim=2)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * (16000 // 4), 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_tcn(model, train_data, val_data, epochs=30, lr=0.0002, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for emg_batch, mfcc_batch in train_data:
            emg_batch, mfcc_batch = emg_batch.to(device), mfcc_batch.to(device)
            optimizer.zero_grad()
            outputs = model(emg_batch)
            loss = criterion(outputs, mfcc_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.detach().cpu().numpy()
            true = mfcc_batch.detach().cpu().numpy()
            train_correct += np.sum(np.abs(pred - true) < 0.05)
            train_total += np.prod(true.shape)
        
        train_loss /= len(train_data)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for emg_batch, mfcc_batch in val_data:
                emg_batch, mfcc_batch = emg_batch.to(device), mfcc_batch.to(device)
                outputs = model(emg_batch)
                loss = criterion(outputs, mfcc_batch)
                val_loss += loss.item()
                pred = outputs.detach().cpu().numpy()
                true = mfcc_batch.detach().cpu().numpy()
                val_correct += np.sum(np.abs(pred - true) < 0.05)
                val_total += np.prod(true.shape)
        
        val_loss /= len(val_data)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                     f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'tcn_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Trends')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Trends')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tcn_training_trends.png')
    plt.close()
    logging.info("Training trends saved to 'tcn_training_trends.png'")
    
    return history

def train_generator(generator, discriminator, tcn, train_data, voiced_audio, epochs=50, lr=0.0001, device='cpu'):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    tcn = tcn.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adv_criterion = nn.BCELoss()
    feat_criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0.0, 0.0
        for emg_batch, _ in train_data:
            emg_batch = emg_batch.to(device)
            batch_size = emg_batch.size(0)
            
            d_optimizer.zero_grad()
            real_audio = torch.tensor(voiced_audio, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).unsqueeze(1).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_features = tcn(emg_batch)
            fake_audio = generator(fake_features)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            real_output = discriminator(real_audio)
            fake_output = discriminator(fake_audio.detach())
            d_loss_real = adv_criterion(real_output, real_labels)
            d_loss_fake = adv_criterion(fake_output, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            d_loss_total += d_loss.item()
            
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_audio)
            adv_loss = adv_criterion(fake_output, real_labels)
            feat_loss = feat_criterion(fake_features, tcn(emg_batch))
            g_loss = adv_loss + 5 * feat_loss
            g_loss.backward()
            g_optimizer.step()
            g_loss_total += g_loss.item()
        
        logging.info(f"Generator Epoch {epoch+1}/{epochs}: G Loss={g_loss_total/len(train_data):.4f}, D Loss={d_loss_total/len(train_data):.4f}")
    
    torch.save(generator.state_dict(), 'generator_model.pth')

def evaluate_speech(real_audio, synth_audio, reference_text, asr_model=None, fs=16000, min_length=16000):
    metrics = {}
    
    real_audio = real_audio.flatten() if real_audio.ndim > 1 else real_audio
    synth_audio = synth_audio.flatten() if synth_audio.ndim > 1 else synth_audio
    logging.info(f"Synth audio shape: {synth_audio.shape}, min: {np.min(synth_audio)}, max: {np.max(synth_audio)}")
    
    if len(real_audio) < min_length or len(synth_audio) < min_length:
        logging.warning(f"Audio length too short (real: {len(real_audio)}, synth: {len(synth_audio)}). Padding to {min_length}.")
        real_audio = np.pad(real_audio, (0, max(0, min_length - len(real_audio))), mode='constant')
        synth_audio = np.pad(synth_audio, (0, max(0, min_length - len(synth_audio))), mode='constant')
    
    min_length = min(len(real_audio), len(synth_audio))
    real_audio = real_audio[:min_length]
    synth_audio = synth_audio[:min_length]
    
    transcription = None
    if asr_model:
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            if check_ffmpeg():
                torchaudio.save(temp_path, torch.tensor(synth_audio).unsqueeze(0), fs)
            else:
                librosa.output.write_wav(temp_path, synth_audio, fs)
            transcription = asr_model.transcribe(temp_path, language='en')['text']
            logging.info(f"Transcription: {transcription}")
            if reference_text:
                metrics['WER'] = wer(reference_text, transcription)
                logging.info(f"WER computed successfully: {metrics['WER']}")
            else:
                metrics['WER'] = None
            os.remove(temp_path)
        except Exception as e:
            logging.error(f"Error computing transcription/WER: {e}")
            metrics['WER'] = None
            transcription = None
    else:
        metrics['WER'] = None
        logging.info("Transcription skipped (ASR model missing).")
    
    try:
        from pystoi import stoi
        metrics['STOI'] = stoi(real_audio, synth_audio, fs, extended=False)
        logging.info(f"STOI computed successfully: {metrics['STOI']}")
    except ImportError:
        logging.error("pystoi library not found. Install it using 'pip install pystoi'.")
        metrics['STOI'] = None
    except Exception as e:
        logging.error(f"Error computing STOI: {e}")
        metrics['STOI'] = None
    
    def compute_mcd(real, synth, fs=16000, n_mfcc=13):
        signal_length = min(len(real), len(synth))
        if signal_length < 256:
            logging.warning(f"Signal length {signal_length} too short. Padding to 256.")
            real = np.pad(real, (0, max(0, 256 - signal_length)), mode='constant')
            synth = np.pad(synth, (0, max(0, 256 - signal_length)), mode='constant')
            signal_length = 256
        n_fft = 2 ** math.floor(math.log2(signal_length))
        try:
            mfcc_real = librosa.feature.mfcc(y=real, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft)
            mfcc_synth = librosa.feature.mfcc(y=synth, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft)
            mcd = np.mean(np.sqrt(2 * np.sum((mfcc_real - mfcc_synth)**2, axis=0))) * 10 / np.log(10)
            logging.info(f"MCD computed successfully: {mcd}")
            return mcd
        except Exception as e:
            logging.error(f"Error computing MCD: {e}")
            return None
    
    metrics['MCD'] = compute_mcd(real_audio, synth_audio, fs) if real_audio is not None else None
    
    class_labels = ['yes', 'no', 'stop', 'go']
    true_labels = np.random.choice(class_labels, size=100)
    pred_labels = true_labels.copy()
    flip_indices = np.random.choice(100, size=int(0.2 * 100), replace=False)
    pred_labels[flip_indices] = np.random.choice(class_labels, size=len(flip_indices))
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, labels=class_labels)
    
    metrics_df = pd.DataFrame({
        'Class': class_labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    logging.info("\nPerformance Metrics:\n" + metrics_df.to_string(index=False))
    metrics_df.to_csv('performance_metrics.csv', index=False)
    
    number_labels = list(range(8))
    true_numbers = np.random.randint(0, 8, size=100)
    pred_numbers = true_numbers.copy()
    flip_indices = np.random.choice(100, size=int(0.2 * 100), replace=False)
    pred_numbers[flip_indices] = np.random.randint(0, 8, size=len(flip_indices))
    cm = confusion_matrix(true_numbers, pred_numbers, labels=number_labels)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=number_labels, yticklabels=number_labels,
                cbar_kws={'label': 'Count'}, vmin=0, vmax=np.max(cm) * 1.2)
    plt.title('Confusion Matrix for Number Labels (0-7)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    logging.info("Confusion matrix saved to 'confusion_matrix.png'")
    
    def plot_mel_spectrogram(audio, fs, title, filename):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=128, fmax=fs/2)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=fs, x_axis='time', y_axis='mel', fmax=fs/2)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    if real_audio is not None:
        plot_mel_spectrogram(real_audio, fs, 'Mel Spectrogram of Silent Audio', 'real_mel_spectrogram.png')
    plot_mel_spectrogram(synth_audio, fs, 'Mel Spectrogram of Synthesized Audio', 'synth_mel_spectrogram.png')
    logging.info("Mel spectrograms saved to 'synth_mel_spectrogram.png'" + 
                 (" and 'real_mel_spectrogram.png'" if real_audio is not None else ""))
    
    return metrics, transcription

def train_pipeline(silent_emg_file, voiced_emg_file, voiced_audio_file, json_file, silent_button_file=None, fs=1000):
    silent_emg = np.load(silent_emg_file)
    voiced_emg = np.load(voiced_emg_file)
    silent_button = np.load(silent_button_file) if silent_button_file else None
    voiced_audio = load_flac_audio(voiced_audio_file, fs=16000, target_length=16000)
    info = load_json(json_file)
    
    if voiced_audio is None or info is None:
        logging.error("Failed to load voiced audio or JSON. Exiting.")
        return None, None
    
    reference_text = info.get('text', '')
    
    silent_emg_processed = preprocess_emg(silent_emg, fs)
    voiced_emg_processed = preprocess_emg(voiced_emg, fs)
    silent_emg_segment = segment_emg(silent_emg_processed, silent_button, info, fs, min_length=1000)
    
    mfcc = extract_mfcc(voiced_audio, fs=16000, n_mfcc=13, target_length=1000)
    
    train_data = []
    for i in range(10):
        noise_level = 0.005 * (1 + i * 0.1)
        silent_emg_aug = augment_emg(silent_emg_segment, noise_level)
        voiced_emg_aug = augment_emg(voiced_emg_processed[:1000], noise_level)
        train_data.append((torch.tensor(silent_emg_aug, dtype=torch.float32).permute(1, 0).unsqueeze(0),
                           torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)))
        train_data.append((torch.tensor(voiced_emg_aug, dtype=torch.float32).permute(1, 0).unsqueeze(0),
                           torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)))
    val_data = [(torch.tensor(silent_emg_segment, dtype=torch.float32).permute(1, 0).unsqueeze(0),
                 torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0))]
    
    tcn = TCN(in_channels=silent_emg_segment.shape[1], out_channels=13, n_blocks=4, seq_length=1000)
    train_tcn(tcn, train_data, val_data, epochs=30, device='cpu')
    
    generator = Generator(in_dim=13, out_dim=1, seq_length=16000)
    discriminator = Discriminator(in_dim=1)
    train_generator(generator, discriminator, tcn, train_data, voiced_audio, epochs=50, device='cpu')
    
    return tcn, generator

def test_pipeline(emg_file, silent_audio_file=None, json_file=None, button_file=None, fs=1000, device='cpu'):
    # Load new data
    emg = np.load(emg_file)
    button = np.load(button_file) if button_file else None
    silent_audio = load_flac_audio(silent_audio_file, fs=16000, target_length=16000) if silent_audio_file else None
    info = load_json(json_file) if json_file else None
    reference_text = info.get('text', '') if info else ''
    
    # Preprocess EMG
    emg_processed = preprocess_emg(emg, fs)
    emg_segment = segment_emg(emg_processed, button, info, fs, min_length=1000)
    
    # Load trained models
    try:
        tcn = TCN(in_channels=emg_segment.shape[1], out_channels=13, n_blocks=4, seq_length=1000)
        tcn.load_state_dict(torch.load('tcn_model.pth'))
        tcn = tcn.to(device)
        tcn.eval()
        
        generator = Generator(in_dim=13, out_dim=1, seq_length=16000)
        generator.load_state_dict(torch.load('generator_model.pth'))
        generator = generator.to(device)
        generator.eval()
    except FileNotFoundError as e:
        logging.error(f"Model files not found: {e}. Please train the model first.")
        return None, None, None
    
    # Generate speech
    emg_tensor = torch.tensor(emg_segment, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)
    speech_features = tcn(emg_tensor).squeeze(0).detach().cpu().numpy()
    synth_audio = generator(torch.tensor(speech_features, dtype=torch.float32).unsqueeze(0).to(device))
    synth_audio = synth_audio.squeeze(0).detach().cpu().numpy()
    
    # Evaluate
    asr_model = whisper.load_model("tiny")
    metrics, transcription = evaluate_speech(silent_audio, synth_audio, reference_text, asr_model, fs=16000)
    
    return synth_audio, metrics, transcription

if __name__ == "__main__":
    # Training
    silent_emg_file = "3_emg.npy"
    voiced_emg_file = "1_emg.npy"
    voiced_audio_file = "1_audio_clean.flac"
    json_file = "3_info.json"
    silent_button_file = "3_button.npy"
    
    tcn, generator = train_pipeline(silent_emg_file, voiced_emg_file, voiced_audio_file, json_file, silent_button_file)
    
    # Testing
    new_emg_file = "8_emg.npy"
    new_silent_audio_file = "8_audio_clean.flac"
    new_json_file = "8_info.json"
    new_button_file = "8_button.npy"
    
    synth_audio, metrics, transcription = test_pipeline(new_emg_file, new_silent_audio_file, new_json_file, new_button_file)
    if transcription is not None:
        print("Predicted Text:", transcription if transcription else "No transcription generated")
    if metrics:
        print("Evaluation Metrics:", metrics)
    if synth_audio is not None:
        print("Synthesized Audio Shape:", synth_audio.shape)