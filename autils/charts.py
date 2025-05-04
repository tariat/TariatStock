"""
    Stock Chart
"""
import FinanceDataReader as fdr
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_candle_charts(tickers, start_date, end_date, ticker_nm_lst=[]):
    """
    주어진 티커 목록과 기간에 대해 세로로 배치된 캔들차트를 생성합니다.
    모든 종목이 하나의 큰 그래프에 세로로 배치됩니다.
    
    Parameters:
    tickers (list): 주식 티커 심볼 리스트 (예: ['005930', '035720'])
    start_date (str): 시작 날짜 (YYYY-MM-DD 형식)
    end_date (str): 종료 날짜 (YYYY-MM-DD 형식)
    
    Returns:
    None: 차트를 화면에 표시합니다.
    """
    # 날짜 형식 확인
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요.")
        return
    
    # 티커 수
    n_tickers = len(tickers)
    
    # 데이터프레임 저장용 딕셔너리
    data_dict = {}
    ticker_names = {}
    
    # 모든 티커의 데이터 먼저 수집
    print("데이터 수집 중...")
    for idx, ticker in enumerate(tickers):
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            if df.empty:
                print(f"{ticker}에 대한 데이터가 없습니다.")
                continue
            data_dict[ticker] = df
            # 종목명 저장 (있는 경우)
            if len(ticker_nm_lst)>0:
                ticker_names[ticker] = ticker_nm_lst[idx]
            else:
                ticker_names[ticker] = ticker
        except Exception as e:
            print(f"{ticker} 데이터 수집 중 오류 발생: {e}")
    
    # 데이터가 없는 경우 종료
    if not data_dict:
        print("표시할 데이터가 없습니다.")
        return
    
    # 서브플롯 생성 (각 티커마다 캔들차트와 거래량 subplot 생성)
    # 각 종목당 2개의 row 필요 (캔들차트 + 거래량)
    fig = make_subplots(
        rows=n_tickers * 2,  # 각 종목마다 캔들차트와 거래량 차트를 위한 두 개의 행
        cols=1,  # 한 열로 모든 차트 배치
        subplot_titles=[f"{ticker} ({ticker_names[ticker]})" for ticker in data_dict.keys()],
        vertical_spacing=0.03,
        shared_xaxes=True,  # 각 종목의 캔들차트와 거래량 차트가 x축 공유
        row_heights=[0.7, 0.3] * n_tickers  # 각 종목마다 캔들차트(0.7)와 거래량(0.3) 비율 설정
    )
    
    # 각 티커에 대해 차트 추가
    row_idx = 1
    for ticker, df in data_dict.items():
        # 캔들차트 추가
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=f'{ticker} 가격'
            ),
            row=row_idx, col=1
        )
        
        # 거래량 바 추가 (다음 행에)
        colors = ['red' if row_data['Close'] < row_data['Open'] else 'green' for _, row_data in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name=f'{ticker} 거래량'
            ),
            row=row_idx + 1, col=1
        )
        
        # y축 타이틀 설정
        fig.update_yaxes(title_text="가격", row=row_idx, col=1)
        fig.update_yaxes(title_text="거래량", row=row_idx + 1, col=1)
        
        # x축 범위슬라이더는 마지막 차트에만 표시
        if row_idx < (n_tickers * 2 - 1):
            fig.update_xaxes(rangeslider_visible=False, row=row_idx, col=1)
            fig.update_xaxes(rangeslider_visible=False, row=row_idx + 1, col=1)
        else:
            fig.update_xaxes(rangeslider_visible=False, row=row_idx, col=1)
            fig.update_xaxes(rangeslider_visible=True, row=row_idx + 1, col=1)
        
        # 다음 종목의 행 인덱스
        row_idx += 2
    
    # 전체 레이아웃 설정
    fig.update_layout(
        title_text=f"주가 차트 ({start_date} ~ {end_date})",
        height=350 * n_tickers,  # 종목 수에 비례하여 높이 설정
        width=1000,   # 너비 고정
        showlegend=False
    )
    
    # 차트 표시
    fig.show()
    print(f"{len(data_dict)}개 종목의 차트가 생성되었습니다.")

if __name__=="__main__":
    # 차트그리기 사용 예시
    # 예시: 삼성전자와 카카오의 2023년 1월부터 2023년 12월까지의 주가 차트
    tickers = ['005930', '035720']  # 삼성전자, 카카오
    start_date = '2023-01-01'
    end_date = '2023-12-31'
        
    create_candle_charts(tickers, start_date, end_date)


