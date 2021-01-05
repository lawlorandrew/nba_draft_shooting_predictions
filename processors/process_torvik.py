import json
import pandas as pd
import os

totals_df = pd.DataFrame()
directory = 'data/draft/torvik'
for filename in os.listdir(directory):
  print(filename)
  with open(os.path.join(directory, filename)) as jsonFile:
    data = json.load(jsonFile)
    stats_df = pd.DataFrame()
    for datum in data:
      row = pd.Series()
      row['Name'] = datum[0]
      row['School'] = datum[1]
      row['Conference'] = datum[2]
      row['G'] = datum[3]
      row['Min%'] = datum[4]
      row['ORtg'] = datum[5]
      row['USG'] = datum[6]
      row['EFG'] = datum[7]
      row['TS%'] = datum[8]
      row['OR%'] = datum[9]
      row['DR%'] = datum[10]
      row['Assist%'] = datum[11]
      row['TO%'] = datum[12]
      row['FT'] = datum[13]
      row['FTA'] = datum[14]
      row['2P'] = datum[16]
      row['2PA'] = datum[17]
      row['3P'] = datum[19]
      row['3PA'] = datum[20]
      row['BLK%'] = datum[22]
      row['STL%'] = datum[23]
      row['Class'] = datum[25]
      row['Height'] = datum[26]
      row['Fouls/40'] = datum[30]
      row['Season'] = datum[31]
      row['PlayerID'] = datum[32]
      row['Close2'] = datum[36]
      row['Close2A'] = datum[37]
      row['Far2'] = datum[38]
      row['Far2A'] = datum[39]
      row['Dunk'] = datum[42]
      row['DunkA'] = datum[43]
      row['Role'] = datum[64]
      stats_df = stats_df.append(row, ignore_index = True)
    totals_df = totals_df.append(stats_df)
  totals_df.to_csv(f'./data/torvik.csv')