import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('src/main/resources/train.csv')

# preprocess data
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# drop data that isn't relevant
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# target features
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
y = data['Survived'].values

# standardize features with scikit-learn
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split training data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Adaline model, batch gradient
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

            # batch gd weight and bias update
            self.weights += self.lr * np.dot(X.T, errors) / samples
            self.bias += self.lr * errors.mean()

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.5, 1, 0)

# train Adaline
adaline = Adaline(learning_rate=0.0001, iters=1000)
adaline.fit(X_train, y_train)

# plot loss curve
plt.plot(range(1, len(adaline.losses) + 1), adaline.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()

# evaluation based on training data
y_train_pred = adaline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# evaluation based on testing data
y_test_pred = adaline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# print results
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
