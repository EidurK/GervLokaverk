import pandas as pd
import numpy as np

import createLogisticModel as clm
import printout as po

from sentence_transformers import SentenceTransformer

model_url = './models/sbertModel.pkl'

submission_data_link = '../data/dataframe.pkl'
df = pd.read_pickle(submission_data_link)

model = SentenceTransformer("all-MiniLM-L6-v2")
po.green(df.columns)
embeddings = model.encode(df['text'])
po.green(embeddings.shape)

