import pandas as pd
from tabulate import tabulate

submission_data_link = '../data/dataframe.pkl'
df = pd.read_pickle(submission_data_link)


def printColumns(results):
    selected_columns = ['text', 'text_split']
    table = tabulate(results[selected_columns], headers='keys', tablefmt='grid')
    print(table)


result = df[df['author'] == '879f283b831c13474e219e88663d95b0763cca9b'] 

print(result.shape)
printColumns(result)


if not result.empty:
    result = ' '.join(result.astype(str).values.flatten())
else:
    result = "No matching rows found"

