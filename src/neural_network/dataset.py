import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class ECGDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        """
        Args:
            data_path (str): Path to the .npy file containing signal data.
            labels_path (str): Path to the .pkl file containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = np.load(data_path)

        with open(labels_path, 'rb') as f:
            self.y_raw = pickle.load(f)

        # Process labels: Convert dicts to multi-hot encoding or single class
        # For this example, let's assume we want to detect if 'MI' (Myocardial Infarction) is present.
        # This is a simplification. In a real project, you'd map all SCP codes to classes.
        self.y = self._process_labels(self.y_raw)

        self.transform = transform

    def _process_labels(self, raw_labels):
        """
        Convert list of dicts to a binary tensor for a specific condition (e.g., MI).
        """
        # Example: Target class 'MI' (Myocardial Infarction)
        # PTB-XL superclasses: NORM, MI, STTC, CD, HYP
        # We need to map the specific SCP codes to these superclasses usually.
        # For this MVP, let's look for 'MI' in the keys or values.

        # NOTE: In a real scenario, you need the 'scp_statements.csv' to map codes to superclasses.
        # Here we will just create dummy labels or look for a specific string if available.
        # Let's create a dummy target: 1 if 'MI' is in the dict keys, 0 otherwise.

        targets = []
        for label_dict in raw_labels:
            # Check if any key contains 'MI' (very naive approach)
            is_mi = 0
            for key in label_dict.keys():
                if 'MI' in key:
                    is_mi = 1
                    break
            targets.append(is_mi)

        return torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Shape: (time, channels) -> PyTorch expects (channels, time) for 1D Conv
        signal = self.X[idx]

        # Transpose to (channels, time)
        signal = signal.transpose(1, 0)

        # Convert to tensor
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = self.y[idx]

        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, label_tensor
