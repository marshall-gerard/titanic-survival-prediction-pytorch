from src.train import train_model
from src.evaluate import evaluate_model
from src.data_loader import preprocess_and_save

if __name__ == '__main__':
    print("01. Preprocessing raw data")
    preprocess_and_save()

    print("02. Training model")
    train_model()

    print("03. Evaluating model")
    evaluate_model()