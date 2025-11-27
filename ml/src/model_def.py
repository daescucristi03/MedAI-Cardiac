"""
Model Definition for MedAI-Cardiac
----------------------------------

This module defines the architecture of the neural network used for
cardiac risk prediction. The chosen model is an MLPClassifier from
scikit-learn, suitable for tabular medical data and interpretable risk scoring.

Architecture:
- Hidden layers: 32 neurons → 16 neurons
- Activation function: ReLU
- Optimizer: Adam
- Iterations: 300 (sufficient for convergence in this dataset)

This architecture offers a good balance between accuracy and training speed
for structured clinical datasets.
"""

from sklearn.neural_network import MLPClassifier


def create_model():
    """
    Creates and returns a configured MLP neural network model.

    Returns:
        MLPClassifier: A neural network with two hidden layers (32, 16)
                       using ReLU activations and Adam optimizer.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )

    return model