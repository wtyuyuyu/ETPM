Code for **Dynamic tactile process sensing on the basis of fingerprint-like sensor array and early temporal perception model**

### 1. ETPM.py

**Overview**<br />
This Python file serves as the backbone for data ingestion and preprocessing, as well as implementing machine learning algorithms for texture classification.<br />

**Dependencies**<br />
Python 3.x, PyTorch, Scikit-learn, NumPy, Pandas, tqdm, argparse

**Methods and Classes**<br />
parse_args()
    This function parses command-line arguments to configure the training process. The parameters include:
        alpha: Coefficient for the early reward loss function.
        epsilon: Threshold for the early stopping reward function.
        learning-rate: Learning rate for the optimizer.
        weight-decay: Weight decay for the optimizer.
        patience: Number of epochs to wait before early stopping if validation loss doesn't improve.
        device: Specifies whether to use GPU or CPU for training.
        epochs: Number of training epochs.
        sequencelength: Length of each input sequence.
        batchsize: Batch size for training.
        snapshot: Path to save the trained model.
        resume: Whether to resume training from a saved model.
