### src/evaluate.py
import torch
from torch.utils.data import DataLoader
from src.models import TitanicModel
from src.dataset import TitanicDataset
from sklearn.metrics import accuracy_score

def evaluate_model():
    test_ds = TitanicDataset('data/processed/train_features.csv', 'data/processed/train_labels.csv')
    test_loader = DataLoader(test_ds, batch_size=32)
    model = TitanicModel(input_dim=test_ds.X.shape[1])
    model.load_state_dict(torch.load('models/titanic_model.pt'))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_batch = y_batch.unsqueeze(1)
            preds = (model(X_batch) > 0.5).float()
            all_preds.extend(preds.numpy().astype(int).flatten())
            all_labels.extend(y_batch.numpy().astype(int).flatten())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Evaluation Accuracy: {acc:.2%}")