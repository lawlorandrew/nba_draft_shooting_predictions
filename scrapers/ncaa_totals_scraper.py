from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

offset = 16000
totals_df = pd.DataFrame()
table_exists = True

chunk = 16
while (table_exists is True):
  try:
    # url = f"https://www.sports-reference.com/cbb/play-index/psl_finder.cgi?request=1&match=single&year_min=2007&year_max=&conf_id=&school_id=&class_is_fr=Y&class_is_so=Y&class_is_jr=Y&class_is_sr=Y&pos_is_g=Y&pos_is_f=Y&pos_is_c=Y&games_type=A&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&order_by=pts&order_by_asc=&offset={offset}"
    url = f"https://www.sports-reference.com/cbb/play-index/psl_finder.cgi?request=1&match=combined&year_min=2007&year_max=2020&conf_id=&school_id=&class_is_fr=Y&class_is_so=Y&class_is_jr=Y&class_is_sr=Y&pos_is_g=Y&pos_is_f=Y&pos_is_c=Y&games_type=A&qual=&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&order_by=pts&order_by_asc=&offset={offset}"
    req = requests.get(url)
    print(offset)
    soup = BeautifulSoup(req.content, "html.parser")
    table = soup.find('table', id='stats')
    table_exists = table is not None
    if (table_exists):
      rows = soup.find('tbody').find_all('tr', class_=lambda x: x is None)
      stats_df = pd.DataFrame()
      for row in rows:
        row_data = pd.Series()
        tds = row.find_all('td')
        for td in tds:
          row_data[td['data-stat']] = td.get_text()
        link = row.find('a', href=lambda value: value and value.startswith("/cbb/players"))
        player_cbb_url = "https://www.sports-reference.com" + link['href']
        player_cbb_req = requests.get(player_cbb_url)
        player_cbb_soup = BeautifulSoup(player_cbb_req.content, 'html.parser')
        nba_link = player_cbb_soup.find('a',href=lambda value: value and value.startswith("https://www.basketball-reference.com/players/"))
        if (nba_link):
          nba_url = nba_link["href"]
          nba_req = requests.get(nba_url)
          nba_soup = BeautifulSoup(nba_req.content, "html.parser")
          totals_table = nba_soup.find("table", id="per_game")
          if (totals_table):
            nba_rows = totals_table.find("tbody").find_all("tr")
            first_row = nba_rows[0]
            first_team = first_row.find('td', {"data-stat":"team_id"})
            i = 0
            while (first_team is None):
              first_team = nba_rows[i].find('td', {"data-stat":"team_id"})
              i += 1
            first_team_text = first_team.get_text()
            if (first_team_text == 'TOT'):
              second_row = nba_rows[1]
              first_real_team = second_row.find('td', {"data-stat":"team_id"})
              first_team_text = first_real_team.get_text()
            row_data["NBA_first_team"] = first_team_text
            footer = totals_table.find("tfoot")
            career_totals = footer.find_all("tr")[0]
            nba_career_tds = career_totals.find_all('td')
            for td in nba_career_tds:
              row_data[f"NBA_{td['data-stat']}"] = td.get_text()
        stats_df = stats_df.append(row_data, ignore_index=True)
      totals_df = totals_df.append(stats_df, ignore_index=True)
      offset += 100
      if (offset % 1000 == 0):
        print(chunk)
        totals_df.to_csv(f'./data/draft/ncaa-totals/ncaa-totals-since-2006-07-chunk-{chunk}.csv')
        chunk += 1
        totals_df = pd.DataFrame()
      time.sleep(2)
  except ConnectionError:
    print('connection error, waiting 20s and trying again')
    time.sleep(20)
totals_df.to_csv(f'./data/draft/ncaa-totals/ncaa-totals-since-2006-07-chunk-{chunk}.csv')
