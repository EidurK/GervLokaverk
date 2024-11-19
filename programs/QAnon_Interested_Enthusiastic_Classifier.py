import pandas as pd
import numpy as np

# Creates and trains logistic regression model and saves it.
# If a model exists it loads it instead
import createLogisticModel as clm
import printout as po

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

model_url = './models/vectorized_model.pkl'
submission_data_link = '../data/dataframe.pkl'
df = pd.read_pickle(submission_data_link)

vectorizer = TfidfVectorizer(max_features=100, analyzer='word', stop_words='english')

X = vectorizer.fit_transform(df['text'])
y = df['isUQ']

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.3,
        random_state=1
        )

clf = clm.get_model(X_train, y_train, model_url)


feature_names = vectorizer.get_feature_names_out()
classes = clf.classes_
coefficients = clf.coef_[0]

top_positive_indices = np.argsort(coefficients)[-10:][::-1]
top_negative_indices = np.argsort(coefficients)[:10]

top_positive_features = [feature_names[i] for i in top_positive_indices]
top_negative_features = [feature_names[i] for i in top_negative_indices]

print("\033[92m","Most common features for QAnon-enthusiastic",top_positive_features,"\033[0m")
print("\033[91m","Most common features for QAnon-interested",top_negative_features,"\033[0m")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

if accuracy > 0.6 : 
    print("Accuracy:","\033[92m",accuracy,"\033[0m")
else:
    print("Accuracy:","\033[91m",accuracy,"\033[0m")



