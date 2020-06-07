import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LogisticRegression
import datetime
import pickle

df = pd.read_csv("Cleansed-Dataset.csv")

df = df[["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing", "Age", "Infection_Prob"]]


def data_split(data, ratio):
    np.random.seed(42)
    train, test = model_selection.train_test_split(data, test_size=ratio)
    return train, test

train_data, test_data = data_split(df, 0.2)

#Train Test Data
X_train = train_data[["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing", "Age"]].to_numpy()
X_test = test_data[["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing", "Age"]].to_numpy()

#Train Test Data
y_train = train_data[["Infection_Prob"]].to_numpy().reshape(253440, )
y_test = test_data[["Infection_Prob"]].to_numpy().reshape(63360, )

#print(y_train)
clf = LogisticRegression()
clf.fit(X_train, y_train)

probablity = clf.predict_proba(X_test)[0][1]
accuracy = clf.score(X_test, y_test)
print(accuracy)