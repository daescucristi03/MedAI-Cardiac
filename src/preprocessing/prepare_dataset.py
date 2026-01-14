import sys
import os

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import wfdb
import pickle
from src.preprocessing.signal_cleaner import SignalCleaner

def load_and_process_data(raw_dir, processed_dir):
    """
    Loads raw ECG data, cleans it, extracts labels, and saves to processed directory.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    cleaner = SignalCleaner()

    processed_data = []
    processed_labels = []

    # List all header files to identify records
    # We walk through the directory because download might create subfolders
    records = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith(".hea"):
                # Get the record path relative to raw_dir or absolute,
                # but wfdb needs the path without extension
                full_path = os.path.join(root, file)
                record_path = os.path.splitext(full_path)[0]
                records.append(record_path)

    print(f"Found {len(records)} records. Processing...")

    for record_path in records:
        try:
            # Read signal and header
            # sampfrom/sampto can be used to limit length, here we take full length
            signals, fields = wfdb.rdsamp(record_path)

            # Apply cleaning
            # signals shape is (time, channels)
            cleaned_signal = cleaner.process(signals)

            # Extract labels from fields['comments']
            # PTB-XL stores SCP codes in comments like "scp_codes: {'NORM': 100, ...}"
            # We need to parse this string.
            labels = {}
            for comment in fields['comments']:
                if comment.startswith('scp_codes:'):
                    # Safe eval or json parse would be better, but format is python dict-like
                    # ex: scp_codes: {'NORM': 100.0, 'LMI': 0.0}
                    code_str = comment.replace('scp_codes:', '').strip()
                    try:
                        import ast
                        labels = ast.literal_eval(code_str)
                    except:
                        print(f"Could not parse labels for {record_path}")

            # For this MVP, let's just store the primary class (highest likelihood)
            # or just the raw dictionary. Let's store the dict.

            processed_data.append(cleaned_signal)
            processed_labels.append(labels)

        except Exception as e:
            print(f"Error processing {record_path}: {e}")

    # Convert to numpy arrays (assuming all signals have same length)
    # PTB-XL signals are usually 10s (5000 samples at 500Hz) or similar.
    # If lengths differ, we might need padding. PTB-XL is consistent.

    try:
        if len(processed_data) == 0:
            print("No data processed. Check if raw data exists.")
            return

        X = np.array(processed_data)
        # y is a list of dicts, we'll save it as an object array or pickle it
        y = np.array(processed_labels)

        print(f"Saving processed data: X shape {X.shape}, y shape {y.shape}")

        np.save(os.path.join(processed_dir, 'X_data.npy'), X)
        # Save labels as pickle because it's a list of dicts
        with open(os.path.join(processed_dir, 'y_labels.pkl'), 'wb') as f:
            pickle.dump(y, f)

        print("Processing complete.")

    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    # Setup paths
    # base_dir is project_root calculated above
    base_dir = project_root
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')

    load_and_process_data(raw_dir, processed_dir)
