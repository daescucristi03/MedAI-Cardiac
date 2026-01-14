import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

# Add project root to sys.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.neural_network.dataset import ECGDataset
from src.neural_network.model import CNNLSTM

def evaluate():
    # Paths
    base_dir = project_root
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    data_path = os.path.join(processed_dir, 'X_data.npy')
    labels_path = os.path.join(processed_dir, 'y_labels.pkl')
    model_path = os.path.join(base_dir, 'src', 'neural_network', 'saved_model.pth')

    # Check files
    if not os.path.exists(model_path):
        print("Model file not found. Train the model first.")
        return

    # 1. Load Data
    dataset = ECGDataset(data_path, labels_path)

    print("WARNING: Evaluating on the full dataset (Train + Val).")

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(input_channels=12, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model loaded. Running predictions...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Convert probabilities to binary predictions
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Check unique classes in predictions and labels
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    print(f"Unique labels in data: {unique_labels}")
    print(f"Unique predictions: {unique_preds}")

    # 3. Metrics
    print("\nClassification Report:")
    # Force labels=[0, 1] to ensure both classes are reported even if one is missing in preds
    # Also handle case where data might only have 0s if dataset is tiny
    target_names = ['Normal/Other', 'MI']

    try:
        print(classification_report(all_labels, all_preds, labels=[0, 1], target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"Could not generate full report: {e}")
        print(classification_report(all_labels, all_preds, zero_division=0))

    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    # Plot
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate()
