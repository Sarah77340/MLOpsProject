import numpy as np

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def test_softmax_properties():
    """La fonction softmax renvoie des proba positives qui somment à 1."""
    vec = np.array([1.0, 2.0, 3.0])
    probs = softmax(vec)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    assert np.all(probs > 0)

def test_emotion_labels_length():
    """On vérifie qu'il y a 7 labels (jeu FER2013)."""
    assert len(emotion_labels) == 7
