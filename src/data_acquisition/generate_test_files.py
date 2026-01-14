import numpy as np
import pandas as pd
import os

def generate_ecg_signal(duration=10, fs=500, heart_rate=60, st_disp=0.0):
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.zeros_like(t)
    rr_interval = 60.0 / heart_rate
    
    # Baseline
    signal += 0.02 * np.sin(2 * np.pi * 0.1 * t)
    
    for i in range(int(duration / rr_interval) + 1):
        t_beat = i * rr_interval
        if t_beat > duration: break
        
        # P-QRS-T Complex
        signal += 0.1 * np.exp(-((t - (t_beat + 0.2))**2) / (2 * 0.015**2)) # P
        signal -= 0.1 * np.exp(-((t - (t_beat + 0.28))**2) / (2 * 0.005**2)) # Q
        signal += 1.0 * np.exp(-((t - (t_beat + 0.3))**2) / (2 * 0.005**2)) # R
        signal -= 0.2 * np.exp(-((t - (t_beat + 0.32))**2) / (2 * 0.005**2)) # S
        
        # ST Segment
        if abs(st_disp) > 0.01:
            st_center = t_beat + 0.4
            signal += st_disp * np.exp(-((t - st_center)**2) / (2 * 0.06**2))
            
        signal += 0.25 * np.exp(-((t - (t_beat + 0.5))**2) / (2 * 0.04**2)) # T

    # Add noise
    signal += np.random.normal(0, 0.02, size=len(t))
    
    # Create 12 leads
    data = {}
    for i in range(12):
        factor = 1.0 if i >= 6 else 0.7
        data[f"Lead_{i+1}"] = (signal * factor) + np.random.normal(0, 0.01, size=len(t))
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Setup output dir
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, 'data', 'test_files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating test files in {output_dir}...")
    
    # 1. Normal Sinus Rhythm
    df_normal = generate_ecg_signal(heart_rate=70, st_disp=0.0)
    df_normal.to_csv(os.path.join(output_dir, 'normal_sinus.csv'), index=False)
    print("- normal_sinus.csv created")
    
    # 2. STEMI (Infarction)
    df_stemi = generate_ecg_signal(heart_rate=80, st_disp=0.4) # High ST elevation
    df_stemi.to_csv(os.path.join(output_dir, 'stemi_infarction.csv'), index=False)
    print("- stemi_infarction.csv created")
    
    # 3. Tachycardia
    df_tachy = generate_ecg_signal(heart_rate=120, st_disp=0.0)
    df_tachy.to_csv(os.path.join(output_dir, 'tachycardia.csv'), index=False)
    print("- tachycardia.csv created")
    
    print("Done.")
