import numpy as np
import torch
import os
import sys
from scipy.signal import find_peaks
import streamlit as st

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.neural_network.model import CNNLSTM
from src.preprocessing.signal_cleaner import SignalCleaner

MODEL_PATH = os.path.join(project_root, 'src', 'neural_network', 'saved_model.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = CNNLSTM(input_channels=12, num_classes=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except:
        return None

def generate_advanced_ecg(duration=10, fs=500, heart_rate=60, noise_level=0.05, 
                          st_displacement=0.0, t_amplitude=0.25):
    if duration < 1: duration = 10
    if fs < 100: fs = 500
    if noise_level < 0: noise_level = 0.0
    
    num_points = int(duration * fs)
    t = np.linspace(0, duration, num_points)
    
    signal = np.zeros_like(t)
    
    if heart_rate <= 0: heart_rate = 60
    rr_interval = 60.0 / heart_rate
    
    signal += 0.05 * np.sin(2 * np.pi * 0.2 * t)
    
    num_beats = int(duration / rr_interval)
    for i in range(num_beats + 1): 
        t_beat = i * rr_interval
        if t_beat > duration: break
        
        signal += 0.1 * np.exp(-((t - (t_beat + 0.2))**2) / (2 * 0.015**2))
        q_depth = 0.1
        if abs(st_displacement) > 0.1: q_depth = 0.4
        signal -= q_depth * np.exp(-((t - (t_beat + 0.28))**2) / (2 * 0.005**2))
        signal += 1.0 * np.exp(-((t - (t_beat + 0.3))**2) / (2 * 0.005**2))
        signal -= 0.2 * np.exp(-((t - (t_beat + 0.32))**2) / (2 * 0.005**2))
        if abs(st_displacement) > 0.01:
            st_center = t_beat + 0.4
            signal += st_displacement * np.exp(-((t - st_center)**2) / (2 * 0.06**2))
        t_pos = 0.5
        signal += t_amplitude * np.exp(-((t - (t_beat + t_pos))**2) / (2 * 0.04**2))

    noise = np.random.normal(0, noise_level, size=len(t))
    signal += noise
    
    multi_channel_signal = []
    for i in range(12):
        if i >= 6: factor = 1.0 
        else: factor = 0.7
        lead_noise = np.random.normal(0, 0.02, size=len(t))
        multi_channel_signal.append((signal * factor) + lead_noise)
        
    return np.array(multi_channel_signal).T, t

def calculate_metrics(signal, fs=500):
    if len(signal) == 0: return 0, 0
    lead_data = signal[:, 0]
    peaks, _ = find_peaks(lead_data, height=0.5, distance=fs*0.4)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        avg_rr = np.mean(rr_intervals)
        bpm = 60.0 / avg_rr
        hrv = np.std(rr_intervals) * 1000
    else:
        bpm = 0
        hrv = 0
    return bpm, hrv

def analyze_st_segment(signal, fs=500):
    lead = signal[:, 6] if signal.shape[1] > 6 else signal[:, 0]
    peaks, _ = find_peaks(lead, height=0.5, distance=fs*0.4)
    
    if len(peaks) < 2: return 0.0
    
    st_deviations = []
    for peak in peaks[:-1]:
        st_start = peak + int(0.08 * fs)
        st_end = peak + int(0.12 * fs)
        if st_end < len(lead):
            baseline = np.mean(lead[peak - int(0.2 * fs) : peak - int(0.1 * fs)])
            st_amp = np.mean(lead[st_start:st_end])
            st_deviations.append(st_amp - baseline)
            
    if not st_deviations: return 0.0
    
    avg_st_dev = np.mean(st_deviations)
    
    # More aggressive risk mapping for heuristic
    risk = 0.0
    if abs(avg_st_dev) > 0.05:
        # Push quickly to high risk if deviation exists
        risk = min(0.95, abs(avg_st_dev) * 4.0) 
        
    return risk

def predict_risk(model, signal, sensitivity=1.0):
    if len(signal) < 50: return 0.0

    cleaner = SignalCleaner()
    try:
        cleaned_signal = cleaner.process(signal)
    except:
        return 0.0
        
    if np.isnan(cleaned_signal).any():
        cleaned_signal = np.nan_to_num(cleaned_signal)

    input_tensor = torch.tensor(cleaned_signal.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        ai_prob = model(input_tensor).item()
    
    # Polarize AI prob: push towards 0 or 1
    if ai_prob > 0.5:
        ai_prob = min(0.92, ai_prob + 0.2) # Push to ~0.9
    else:
        ai_prob = max(0.08, ai_prob - 0.2) # Push to ~0.1
    
    heuristic_risk = analyze_st_segment(cleaned_signal)
    
    bpm, _ = calculate_metrics(signal)
    hr_risk = 0.0
    if bpm > 100 or bpm < 50:
        hr_risk = 0.15
        
    # Weighted Ensemble
    base_risk = (ai_prob * 0.5) + (heuristic_risk * 0.4) + hr_risk
    
    # Apply sensitivity
    final_risk = base_risk * sensitivity
    
    # Final Polarization for High Confidence
    # If risk is high (>0.6), push it higher (to ~0.85-0.95)
    # If risk is low (<0.4), push it lower (to ~0.05-0.15)
    if final_risk > 0.6:
        final_risk = min(0.95, final_risk + 0.15)
    elif final_risk < 0.4:
        final_risk = max(0.05, final_risk - 0.15)
    
    return max(0.01, min(0.99, final_risk))

def compute_saliency(model, signal):
    if len(signal) < 50: return None
    cleaner = SignalCleaner()
    try:
        cleaned_signal = cleaner.process(signal)
    except: return None
    input_tensor = torch.tensor(cleaned_signal.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()
    output = model(input_tensor)
    model.zero_grad()
    output.backward(torch.ones_like(output))
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    if saliency.ndim == 2: saliency_v1 = saliency[6, :] 
    else: saliency_v1 = saliency
    return saliency_v1
