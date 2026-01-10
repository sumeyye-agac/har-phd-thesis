"""
Evaluation utilities for TFLite models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def evaluate_tflite_model(interpreter, X, y):
    """
    Returns: accuracy, f1, precision, recall, confusion_matrix
    """
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    preds = []

    for sample in X:
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        preds.append(np.argmax(output[0]))

    preds = np.array(preds)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    prec = precision_score(y, preds, average="macro")
    rec = recall_score(y, preds, average="macro")
    cm = confusion_matrix(y, preds)

    return acc, f1, prec, rec, cm