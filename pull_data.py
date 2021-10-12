# 데이터 수집 및 저장

from io import BytesIO
from pandas.io.html import read_html
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

url = 'https://finance.naver.com/item/sise_day.naver?code=005380' # 현대자동차그룹 주가(네이버 증권)
# 마지막 페이지 확인하기
with urlopen(url) as doc:
    req = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}) #url접근을 위한 user-agent 입력
    soup = BeautifulSoup(req.text, 'lxml')
    pgrr = soup.find('td', class_='pgRR')
    s = str(pgrr.a['href']).split('=')
    last_page = s[-1]
#마지막 페이지: 636 

def price_list(page_no):
    url = f'https://finance.naver.com/item/sise_day.naver?code=005380&page={page_no}'
    response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
    html = BeautifulSoup(response.text, 'lxml')
    table = html.select('table')
    table = pd.read_html(str(table))
    df = table[0].dropna()
    return df

HMG_df = pd.DataFrame()

for page in range(1, 636):
    HMG_df = HMG_df.append(price_list(page), ignore_index=True)

HMG_df = HMG_df.dropna()
HMG_df['날짜'] = pd.to_datetime(HMG_df['날짜'])
HMG_df = HMG_df.sort_values(by=['날짜'], ascending=True)

con = sqlite3.connect("HMG.db")

HMG_df.to_sql('Daily_Price', con, index=False)