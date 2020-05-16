import math
import requests
import numpy as np
import pandas as pd
from bs4 import Comment
from selenium import webdriver
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

df = pd.DataFrame()
address = 'r'C:\\Users\\TomTu\\OneDrive - University of Maryland\\2020 Spring\\CMSC 320\\\nba_data.csv'

for year in range(2000,2020):
    url = 'https://www.basketball-reference.com/leagues/NBA_{}.html#all_team-stats-base'.format(str(year))
    page = requests.get(url)

    soup = BeautifulSoup(page.text, 'html.parser')

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    tables = []
    for each in comments:
        if 'table' in each:
            try:
                tables.append(pd.read_html(each)[0])
            except:
                continue
                
    tmp = tables[7]
    tmp.columns = tmp.columns.droplevel()
    tmp.drop(tmp[tmp['Team'] == 'League Average'].index, inplace = True)
    tmp.insert(1,'yearID',str(year))
    del tmp['Rk']
    del tmp['Arena']
    del tmp['Attend.']
    del tmp['Attend./G']
    
    tmp2 = tables[3]
    tmp2.drop(tmp2[tmp2['Team'] == 'League Average'].index, inplace = True)
    
    for i in range(2,len(tmp2.columns)):
        attribute = tmp2.columns[i]
        tmp[attribute] = tmp2[attribute]
    
    df = df.append(tmp)

name = [i.replace('*','') if '*' in i else i for i in df.Team.tolist()]
new_df = pd.DataFrame({'Team': name})

df.update(new_df)
df.head(10)

df.to_csv(address, index = False)
