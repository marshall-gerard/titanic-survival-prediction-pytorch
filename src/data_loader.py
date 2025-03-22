import pandas as pd
import os

def load_raw_data():
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    return train, test

def preprocess_and_save():
    train, test = load_raw_data()

    for df in [train, test]:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 
                                           'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
        df['Title'] = df['Title'].fillna(4)

    train['Age'] = train['Age'].fillna(train['Age'].median())
    test['Age'] = test['Age'].fillna(test['Age'].median())
    train['Fare'] = train['Fare'].fillna(train['Fare'].median())
    test['Fare'] = test['Fare'].fillna(test['Fare'].median())
    train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
    test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])

    for df in [train, test]:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        df['FamilySize'] = df['SibSp'] + df['Parch']
        df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
        df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=[0,1,2,3,4])
        df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0,1,2,3])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin']
    train_features = train[features]
    train_labels = train['Survived']
    test_features = test[features]

    os.makedirs('data/processed', exist_ok=True)
    train_features.to_csv('data/processed/train_features.csv', index=False)
    train_labels.to_csv('data/processed/train_labels.csv', index=False)
    test_features.to_csv('data/processed/test_features.csv', index=False)