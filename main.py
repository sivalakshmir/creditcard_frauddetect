import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    # Load the dataset from the local folder
    path = "creditcard.csv"
    credit_card_data = pd.read_csv(path)
    
    # Separate legitimate and fraudulent transactions
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    # Balance the dataset by under-sampling
    legit_sample = legit.sample(n=492, random_state=1)
    new_dataset = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)
    
    # Split into features and labels
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
    return X, Y

def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    accuracy = accuracy_score(Y_test, model.predict(X_test))
    return model, accuracy

