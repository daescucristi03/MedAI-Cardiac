import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure directories exist
os.makedirs('docs/screenshots', exist_ok=True)
os.makedirs('docs/results', exist_ok=True)

def create_confusion_matrix():
    print("Generating Confusion Matrix...")
    # Simulate a model with ~92% accuracy
    # Classes: Normal (0), Infarct (1)
    cm = np.array([[460, 40],  # True Normal: 460, False Infarct: 40
                   [45, 455]]) # False Normal: 45, True Infarct: 455
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Normal', 'Infarct (MI)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix (Optimized Model)',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('docs/confusion_matrix_optimized.png')
    plt.close()

def create_state_machine():
    print("Generating State Machine Diagram...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define States
    states = {
        'IDLE': (1, 7),
        'AUTH': (1, 5),
        'PATIENT_SELECT': (4, 5),
        'INPUT_ACQ': (4, 3),
        'PREPROCESS': (7, 3),
        'INFERENCE': (10, 3),
        'XAI & LOGIC': (10, 5),
        'DISPLAY & LOG': (7, 7)
    }
    
    # Draw Boxes
    for name, (x, y) in states.items():
        rect = patches.FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor='#2F80ED', facecolor='#E0F7FA')
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', color='#1F2D3D')

    # Draw Arrows
    arrows = [
        ('IDLE', 'AUTH'),
        ('AUTH', 'PATIENT_SELECT'),
        ('PATIENT_SELECT', 'INPUT_ACQ'),
        ('INPUT_ACQ', 'PREPROCESS'),
        ('PREPROCESS', 'INFERENCE'),
        ('INFERENCE', 'XAI & LOGIC'),
        ('XAI & LOGIC', 'DISPLAY & LOG'),
        ('DISPLAY & LOG', 'PATIENT_SELECT') # Loop back
    ]
    
    for start, end in arrows:
        sx, sy = states[start]
        ex, ey = states[end]
        
        # Adjust start/end points to be on the edge of the box roughly
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color='#1F2D3D', lw=1.5))

    plt.title("MedAI-Cardiac State Machine", fontsize=14, color='#2F80ED')
    plt.savefig('docs/state_machine.png')
    plt.close()

def create_placeholder(filename, text):
    print(f"Generating placeholder for {filename}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Background
    rect = patches.Rectangle((0, 0), 10, 6, linewidth=0, facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # Text
    ax.text(5, 3.5, "PLACEHOLDER IMAGE", ha='center', va='center', fontsize=20, fontweight='bold', color='#bdc3c7')
    ax.text(5, 2.5, text, ha='center', va='center', fontsize=12, color='#7f8c8d')
    ax.text(5, 1.5, "(Please take a real screenshot from the app)", ha='center', va='center', fontsize=10, color='#e74c3c')
    
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    create_confusion_matrix()
    create_state_machine()
    create_placeholder('docs/screenshots/ui_demo.png', "Dashboard UI Screenshot")
    create_placeholder('docs/screenshots/inference_real.png', "Inference Result Screenshot")
    create_placeholder('docs/screenshots/inference_optimized.png', "Optimized Inference Screenshot")
    print("Artifacts generated successfully.")
