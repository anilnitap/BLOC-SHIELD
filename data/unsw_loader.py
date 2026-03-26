
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_unsw(path):
    df = pd.read_csv(path)

    # Keep numeric columns only
    df = df.select_dtypes(include=['int64', 'float64'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)
