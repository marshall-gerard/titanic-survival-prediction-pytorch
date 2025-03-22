import torch
from torch.utils.data import DataLoader
from src.models import TitanicModel
from src.dataset import TitanicDataset
from torch import nn, optim
import os

def train_model():
    dataset = TitanicDataset('data/processed/train_features.csv', 'data/processed/train_labels.csv')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = TitanicModel(input_dim=dataset.X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            y_batch = y_batch.unsqueeze(1)
            pred = model(X_batch)
            loss = criterion(pred.squeeze(), y_batch.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation Accuracy
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_batch = y_batch.unsqueeze(1)
                pred = (model(X_batch) > 0.5).float()
                val_correct += (pred == y_batch).sum().item()
                val_total += y_batch.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.2%}")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/titanic_model.pt')