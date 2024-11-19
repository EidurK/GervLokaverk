import os
import joblib
import printout as po
from sklearn.linear_model import LogisticRegression

def does_model_exist(url):
    return os.path.exists(url)

def train_new_model(X_train, y_train):
    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, y_train)
    return clf

def save_model(clf, url):
    joblib.dump(clf, url)
    return clf

def train_and_save_model(X_train, y_train, url='./models/model.pkl'):
    return save_model(train_new_model(X_train, y_train), url)

def load_model(url='./models/model.pkl'):
    return joblib.load(url)

def get_model(X_train, y_train, url):
    if does_model_exist(url):
        po.green("Model successfully loaded!")
        return load_model(url)
    po.red("Model does not exist. Training model...")


    return train_and_save_model(X_train, y_train, url)

