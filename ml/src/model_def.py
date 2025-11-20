from sklearn.neural_network import MLPClassifier

def create_model():
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )
    return model