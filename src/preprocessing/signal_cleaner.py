import numpy as np
from scipy import signal

class SignalCleaner:
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate

    def remove_baseline_wander(self, ecg_signal):
        """
        Removes baseline wander using a high-pass filter.
        """
        cutoff = 0.5
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(1, normal_cutoff, btype='high', analog=False)
        
        if ecg_signal.ndim == 1:
            cleaned_signal = signal.filtfilt(b, a, ecg_signal)
        else:
            cleaned_signal = signal.filtfilt(b, a, ecg_signal, axis=0)
            
        return cleaned_signal

    def remove_powerline_interference(self, ecg_signal, freq=50.0):
        """
        Removes powerline interference (50Hz/60Hz) using a notch filter.
        """
        nyquist = 0.5 * self.sampling_rate
        notch_freq = freq / nyquist
        quality_factor = 30.0
        b, a = signal.iirnotch(notch_freq, quality_factor)
        
        if ecg_signal.ndim == 1:
            cleaned_signal = signal.filtfilt(b, a, ecg_signal)
        else:
            cleaned_signal = signal.filtfilt(b, a, ecg_signal, axis=0)
            
        return cleaned_signal

    def z_score_normalization(self, ecg_signal):
        """
        Standard Z-score normalization.
        """
        mean = np.mean(ecg_signal, axis=0)
        std = np.std(ecg_signal, axis=0)
        std = np.where(std == 0, 1, std)
        return (ecg_signal - mean) / std

    def process(self, ecg_signal):
        # 1. Remove baseline wander
        no_wander = self.remove_baseline_wander(ecg_signal)
        
        # 2. Remove powerline interference (50Hz)
        no_powerline = self.remove_powerline_interference(no_wander, freq=50.0)
        
        # 3. Normalize
        # We use Z-score which is standard for DL models
        normalized = self.z_score_normalization(no_powerline)

        return normalized
