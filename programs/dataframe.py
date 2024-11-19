import pandas as pd
import re
import GloVe

def fix_text(text):
    text = text.lower()
    text = re.sub(r"[!@#~$%^&*()\[\]\\|{}+_\-=/?,.><;:]",'',text)
    return text.split()

submission_data_link = '../data/Hashed_Q_Submissions_Raw_Combined.csv'
user_data_link = '../data/Hashed_allAuthorStatus.csv'

print("GloVe initialization", end='...', flush=True)
glove = GloVe.GloVe()
print("\033[32mcompleted\033[0m")

print("Reading files", end='...', flush=True)
df_submissions = pd.read_csv(submission_data_link)
df_user = pd.read_csv(user_data_link)
print("\033[32mcompleted\033[0m")

print("Removing reduntant columns", end='...', flush=True)
df_submissions = df_submissions.get(['subreddit','score','title', 'text','author','date_created'])

print("\033[32mcompleted\033[0m")

print("Creating additional columns", end='...', flush=True)
df = df_submissions.merge(df_user[['QAuthor', 'isUQ']], left_on='author', right_on='QAuthor', how='left')
df = df.drop(columns=['QAuthor'])
df = df.dropna()
df['text_split'] =df['text'].apply(fix_text)
df['sentence_embedding'] = df['text_split'].apply(glove.average_post_vector)
df = df.dropna()

print("\033[32mcompleted\033[0m")


# df will be df_submissions with an extra column for isUQ
df.to_pickle("../data/dataframe.pkl")
