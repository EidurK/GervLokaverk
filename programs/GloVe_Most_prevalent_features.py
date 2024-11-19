import pandas as pd
import numpy as np

import createLogisticModel as clm
import printout as po
import GloVe 

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import accuracy_score

glv = GloVe.GloVe()
model_url = './models/gloveModel.pkl'

submission_data_link = '../data/dataframe.pkl'
df = pd.read_pickle(submission_data_link)
X = np.vstack(df['sentence_embedding'])
y = df['isUQ']

X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=0.3,
        random_state=1
        )

E = np.array(glv.embeddings)

clf = clm.get_model(X_train, y_train, url=model_url)


distances = cosine_similarity(E, clf.coef_[0].reshape(1,-1)).flatten()

top_positive_indices = np.argsort(distances)[-10:][::-1]
top_negative_indices = np.argsort(distances)[:10]

top_positive_features = [glv.indx2word[i] for i in top_positive_indices]
top_negative_features = [glv.indx2word[i] for i in top_negative_indices]

print("\033[92m","Most common features for QAnon-enthusiastic",top_positive_features,"\033[0m")
print("\033[91m","Most common features for QAnon-interested",top_negative_features,"\033[0m")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

if accuracy > 0.6 : 
    print("Accuracy:","\033[92m",accuracy,"\033[0m")
else:
    print("Accuracy:","\033[91m",accuracy,"\033[0m")

