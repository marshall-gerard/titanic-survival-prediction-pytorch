# Titanic Survival PyTorch

A clean, modular, and professional machine learning pipeline to predict passenger survival on the Titanic using PyTorch, featuring advanced feature engineering and modular code design.

---

## 📂 Project Structure

titanic-survival-pytorch/ ├── data/ │ ├── raw/ ← Original Kaggle CSV files │ └── processed/ ← Generated processed features (ignored) ├── models/ ← Trained models (ignored) ├── src/ │ ├── init.py │ ├── data_loader.py ← Data preprocessing and feature engineering │ ├── dataset.py ← PyTorch Dataset class │ ├── models.py ← PyTorch model definition │ ├── train.py ← Model training loop with validation accuracy │ └── evaluate.py ← Model evaluation (accuracy, precision, recall, F1) ├── main.py ← Runs preprocessing, training, and evaluation ├── requirements.txt ← Required Python packages ├── .gitignore ← Files excluded from GitHub └── README.md ← Project overview


---

## 🚀 Project Overview

This project predicts whether Titanic passengers survived or not based on features such as age, gender, class, fare, family size, and derived features. The model is built using PyTorch and trained with advanced feature engineering to achieve approximately **80% accuracy**.

---

## 📊 Dataset

- [Titanic Kaggle Dataset](https://www.kaggle.com/competitions/titanic/data)

Place the dataset files in the following directory structure:


---

## ⚙️ Installation and Setup

**Clone this repository:**

```bash
git clone https://github.com/marshall-gerard/titanic-survival-pytorch.git
cd titanic-survival-pytorch

pip install -r requirements.txt