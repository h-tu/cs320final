```python
import math
import requests
import numpy as np
import pandas as pd
from bs4 import Comment
from selenium import webdriver
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
```


```python
df = pd.DataFrame()

for year in range(2000,2020):
    #NBA season
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
                
    curr = tables[7]
    curr.columns = curr.columns.droplevel()
    curr.drop(curr[curr['Team'] == 'League Average'].index, inplace = True)
    curr.insert(1,'yearID',str(year))
    del curr['Rk']
    del curr['Arena']
    del curr['Attend.']
    del curr['Attend./G']
    df = df.append(curr)
```


```python
name = [i.replace('*','') if '*' in i else i for i in df.Team.tolist()]
new_df = pd.DataFrame({'Team': name})
```


```python
df.update(new_df)
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yearID</th>
      <th>Team</th>
      <th>Age</th>
      <th>W</th>
      <th>L</th>
      <th>PW</th>
      <th>PL</th>
      <th>MOV</th>
      <th>SOS</th>
      <th>SRS</th>
      <th>...</th>
      <th>3PAr</th>
      <th>TS%</th>
      <th>eFG%</th>
      <th>TOV%</th>
      <th>ORB%</th>
      <th>FT/FGA</th>
      <th>eFG%</th>
      <th>TOV%</th>
      <th>DRB%</th>
      <th>FT/FGA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Los Angeles Lakers</td>
      <td>29.2</td>
      <td>67.0</td>
      <td>15.0</td>
      <td>64</td>
      <td>18</td>
      <td>8.55</td>
      <td>-0.14</td>
      <td>8.41</td>
      <td>...</td>
      <td>0.153</td>
      <td>0.525</td>
      <td>0.484</td>
      <td>12.7</td>
      <td>30.6</td>
      <td>0.241</td>
      <td>0.443</td>
      <td>13.4</td>
      <td>73.1</td>
      <td>0.222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>Portland Trail Blazers</td>
      <td>29.6</td>
      <td>59.0</td>
      <td>23.0</td>
      <td>59</td>
      <td>23</td>
      <td>6.40</td>
      <td>-0.04</td>
      <td>6.36</td>
      <td>...</td>
      <td>0.175</td>
      <td>0.546</td>
      <td>0.501</td>
      <td>14.5</td>
      <td>30.3</td>
      <td>0.240</td>
      <td>0.461</td>
      <td>13.8</td>
      <td>72.4</td>
      <td>0.217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>San Antonio Spurs</td>
      <td>30.9</td>
      <td>53.0</td>
      <td>29.0</td>
      <td>58</td>
      <td>24</td>
      <td>5.94</td>
      <td>-0.02</td>
      <td>5.92</td>
      <td>...</td>
      <td>0.138</td>
      <td>0.535</td>
      <td>0.488</td>
      <td>14.3</td>
      <td>27.8</td>
      <td>0.258</td>
      <td>0.451</td>
      <td>13.5</td>
      <td>73.0</td>
      <td>0.188</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>Phoenix Suns</td>
      <td>28.6</td>
      <td>53.0</td>
      <td>29.0</td>
      <td>56</td>
      <td>26</td>
      <td>5.22</td>
      <td>0.02</td>
      <td>5.24</td>
      <td>...</td>
      <td>0.184</td>
      <td>0.532</td>
      <td>0.491</td>
      <td>15.2</td>
      <td>29.3</td>
      <td>0.217</td>
      <td>0.454</td>
      <td>15.7</td>
      <td>70.5</td>
      <td>0.245</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>Utah Jazz</td>
      <td>31.5</td>
      <td>55.0</td>
      <td>27.0</td>
      <td>54</td>
      <td>28</td>
      <td>4.46</td>
      <td>0.05</td>
      <td>4.52</td>
      <td>...</td>
      <td>0.134</td>
      <td>0.540</td>
      <td>0.490</td>
      <td>14.3</td>
      <td>29.5</td>
      <td>0.260</td>
      <td>0.477</td>
      <td>15.0</td>
      <td>73.2</td>
      <td>0.256</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2000</td>
      <td>Indiana Pacers</td>
      <td>30.4</td>
      <td>56.0</td>
      <td>26.0</td>
      <td>54</td>
      <td>28</td>
      <td>4.60</td>
      <td>-0.45</td>
      <td>4.15</td>
      <td>...</td>
      <td>0.224</td>
      <td>0.552</td>
      <td>0.503</td>
      <td>13.3</td>
      <td>24.9</td>
      <td>0.245</td>
      <td>0.469</td>
      <td>12.6</td>
      <td>71.5</td>
      <td>0.197</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2000</td>
      <td>Sacramento Kings</td>
      <td>27.5</td>
      <td>44.0</td>
      <td>38.0</td>
      <td>49</td>
      <td>33</td>
      <td>2.91</td>
      <td>0.12</td>
      <td>3.04</td>
      <td>...</td>
      <td>0.227</td>
      <td>0.526</td>
      <td>0.486</td>
      <td>13.9</td>
      <td>27.7</td>
      <td>0.209</td>
      <td>0.479</td>
      <td>15.1</td>
      <td>69.7</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2000</td>
      <td>Miami Heat</td>
      <td>29.2</td>
      <td>52.0</td>
      <td>30.0</td>
      <td>50</td>
      <td>32</td>
      <td>3.11</td>
      <td>-0.36</td>
      <td>2.75</td>
      <td>...</td>
      <td>0.186</td>
      <td>0.533</td>
      <td>0.495</td>
      <td>14.5</td>
      <td>28.2</td>
      <td>0.208</td>
      <td>0.453</td>
      <td>13.3</td>
      <td>72.9</td>
      <td>0.229</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2000</td>
      <td>Minnesota Timberwolves</td>
      <td>26.6</td>
      <td>50.0</td>
      <td>32.0</td>
      <td>48</td>
      <td>34</td>
      <td>2.52</td>
      <td>0.14</td>
      <td>2.67</td>
      <td>...</td>
      <td>0.104</td>
      <td>0.525</td>
      <td>0.485</td>
      <td>12.9</td>
      <td>29.4</td>
      <td>0.200</td>
      <td>0.474</td>
      <td>14.0</td>
      <td>73.0</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2000</td>
      <td>Charlotte Hornets</td>
      <td>28.7</td>
      <td>49.0</td>
      <td>33.0</td>
      <td>49</td>
      <td>33</td>
      <td>2.67</td>
      <td>-0.34</td>
      <td>2.33</td>
      <td>...</td>
      <td>0.153</td>
      <td>0.530</td>
      <td>0.475</td>
      <td>13.7</td>
      <td>25.7</td>
      <td>0.285</td>
      <td>0.478</td>
      <td>14.6</td>
      <td>73.2</td>
      <td>0.198</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 25 columns</p>
</div>




```python
df.to_csv(r'C:\\Users\\TomTu\\OneDrive - University of Maryland\\2020 Spring\\CMSC 320\\\nba_data.csv', index = False)
```


```python

```
