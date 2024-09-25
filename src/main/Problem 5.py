import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('src/main/resources/train.csv')

# pre process data
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# drop irrelevant data
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# target features
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
y = data['Survived'].values

# standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# adaline model BGD
class Adaline:
    def __init__(self, learning_rate=0.01, iters=1000):
        self.lr = learning_rate
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0
        self.losses = []

        for _ in range(self.iters):
            linear_output = np.dot(X, self.weights) + self.bias
            errors = y - linear_output
            loss = (errors**2).mean() / 2
            self.losses.append(loss)

            # BGD weight and bias update
            self.weights += self.lr * np.dot(X.T, errors) / samples
            self.bias += self.lr * errors.mean()

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.5, 1, 0)


# train adaline
adaline = Adaline(learning_rate=0.01, iters=1000)
adaline.fit(X_train, y_train)

# predictions
y_train_pred_adaline = adaline.predict(X_train)
y_test_pred_adaline = adaline.predict(X_test)

# find accuracy
train_accuracy_adaline = accuracy_score(y_train, y_train_pred_adaline)
test_accuracy_adaline = accuracy_score(y_test, y_test_pred_adaline)


# random weight baseline model
class BaselineModel:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X):
        samples, features = X.shape
        # randomly initialize weights and bias
        np.random.seed(42)
        self.weights = np.random.randn(features)
        self.bias = np.random.randn(1)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.5, 1, 0)


# train baseline model
baseline_model = BaselineModel()
baseline_model.fit(X_train)

# predict
y_train_pred_baseline = baseline_model.predict(X_train)
y_test_pred_baseline = baseline_model.predict(X_test)

# check performance
train_accuracy_baseline = accuracy_score(y_train, y_train_pred_baseline)
test_accuracy_baseline = accuracy_score(y_test, y_test_pred_baseline)

# compare performance
print(f"Adaline Model Training Accuracy: {train_accuracy_adaline * 100:.2f}%")
print(f"Adaline Model Test Accuracy: {test_accuracy_adaline * 100:.2f}%\n")

print(f"Baseline Model Training Accuracy: {train_accuracy_baseline * 100:.2f}%")
print(f"Baseline Model Test Accuracy: {test_accuracy_baseline * 100:.2f}%")

