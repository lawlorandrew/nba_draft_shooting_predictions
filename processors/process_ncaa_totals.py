import pandas as pd
import os

totals_df = pd.DataFrame()
directory = 'data/draft/ncaa-totals'
for filename in os.listdir(directory):
  print(filename)
  chunk_df = pd.read_csv(os.path.join(directory, filename), index_col='Unnamed: 0')
  print(chunk_df.columns)
  totals_df = totals_df.append(chunk_df, ignore_index=True)
totals_df.to_csv('data/draft/ncaa-totals-2006-07.csv')