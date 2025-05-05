import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from io import StringIO
from tqdm import tqdm

# BASE_DIR에 데이터를 저장할 경로를 지정하세요.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autils import data_dir
from autils.db import SqLite
db = SqLite(f"{data_dir}/stock_5m_data.db")

# SQLite 데이터베이스 테이블 생성
def init_db():
    """
    SQLite 데이터베이스와 테이블을 초기화합니다.
    """

    # 주식 5분봉 데이터를 저장할 테이블 생성
    db.execute('''
    CREATE TABLE IF NOT EXISTS stock_5m_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        trading_value REAL,
        UNIQUE(code, date, time)
    )
    ''')

def crawl_date_by_date(code, date):
    """
    date: yyyymmdd
    """
    url = f"http://finance.naver.com/item/sise_time.nhn?code={code}&thistime={date}180000&page=1"

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")    
    temp = pd.read_html(StringIO(soup.decode_contents()))[0]
    temp.dropna(inplace=True)

    last_page_num = soup.select(".pgRR")[0].select("a")[0].get_attribute_list("href")[0].split("=")[-1]
    last_page_num = int(last_page_num)
    df_lst = list()
    df_lst.append(temp)

    def get_table_by_page(code, date, page):
        url = f"http://finance.naver.com/item/sise_time.nhn?code={code}&thistime={date}180000&page={page}"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        temp = pd.read_html(StringIO(soup.decode_contents()))[0]
        temp.dropna(inplace=True)
        time.sleep(1)

        return temp
    
    for p in tqdm(range(2, last_page_num+1)):  # 여기 range로 수정했습니다
        df_lst.append(get_table_by_page(code, date, p))

    total = pd.concat(df_lst, axis=0)
    total['체결시각'] = date + " " + total['체결시각']

    return total

def create_5min_candle(df):
    # 체결시각 컬럼을 datetime 형식으로 변환
    df['체결시각'] = pd.to_datetime(df['체결시각'], format='%Y%m%d %H:%M')
    
    # 인덱스를 리셋하고 체결시각을 인덱스로 설정
    df = df.reset_index(drop=True).set_index('체결시각')
    df['거래대금'] = df['변동량'] * df['체결가']
    
    # 5분 간격으로 리샘플링
    resampled = pd.DataFrame()
    
    # 시가 (첫 체결가)
    resampled['시가'] = df['체결가'].resample('5min').first()
    
    # 고가 (최대 체결가)
    resampled['고가'] = df['체결가'].resample('5min').max()
    
    # 저가 (최소 체결가)
    resampled['저가'] = df['체결가'].resample('5min').min()
    
    # 종가 (마지막 체결가)
    resampled['종가'] = df['체결가'].resample('5min').last()
    
    # 거래량 (거래량 차이)
    resampled['거래량'] = df['거래량'].resample('5min').first() - df['거래량'].resample('5min').last()
    resampled['거래량'] = resampled['거래량'].abs()  # 양수로 변환
    
    # 거래대금
    # 거래대금 = 거래량 * 평균가격
    resampled['거래대금'] = df['거래대금'].resample('5min').sum()
    
    # 결측값 처리
    resampled = resampled.ffill()
    
    # 인덱스를 컬럼으로 변환하여 반환
    resampled = resampled.reset_index()
    
    return resampled

def save_to_db(code, date, df):
    """
    5분봉 데이터를 SQLite DB에 저장합니다.
    """

    for _, row in df.iterrows():
        time_str = row['체결시각'].strftime('%H:%M')
        date_str = row['체결시각'].strftime('%Y%m%d')
        
        db.execute('''
        INSERT OR REPLACE INTO stock_5m_data 
        (code, date, time, open, high, low, close, volume, trading_value) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            code, 
            date_str,
            time_str, 
            row['시가'], 
            row['고가'], 
            row['저가'], 
            row['종가'], 
            row['거래량'], 
            row['거래대금']
        ))

def get_5m_from_db(code, date):
    """
    DB에서 특정 종목, 특정 날짜의 5분봉 데이터를 조회합니다.
    """
    query = '''
    SELECT time, open, high, low, close, volume, trading_value 
    FROM stock_5m_data 
    WHERE code = ? AND date = ? 
    ORDER BY time
    '''
    
    df = db.get(query, params=(code, date))
    
    if not df.empty:
        # 시간 데이터 형식 조정
        df['time'] = pd.to_datetime(date + ' ' + df['time'])
        df = df.set_index('time')

    return df

def get_5m_by_date(code, date):
    """
    특정 종목, 특정 날짜의 5분봉 데이터를 가져옵니다.
    DB에 있으면 DB에서 가져오고, 없으면 크롤링 후 DB에 저장합니다.
    """
    # 먼저 DB에서 데이터 조회
    df = get_5m_from_db(code, date)
    
    # DB에 데이터가 없으면 크롤링하여 저장
    if df.empty:
        raw_df = crawl_date_by_date(code, date)
        df = create_5min_candle(raw_df)
        save_to_db(code, date, df)
        df = df.rename(columns = {
            "체결시각":"time",
            "시가":"open",
            "고가":"high",
            "저가":"low",
            "종가":"close",
            "거래량":"volume",
            "거래대금":"trading_value"})
    
    return df

if __name__=="__main__":
    # DB 초기화
    init_db()
    
    # 데이터 가져오기
    df = get_5m_by_date("005930", "20250428")
    print(df.head())
    