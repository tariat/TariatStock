"""
    get historical data
"""
import FinanceDataReader as fdr
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_stock_return(ticker, period):
    """
    종목 코드와 기간을 입력받아 해당 기간의 수익률을 계산하는 함수
    
    Parameters:
    -----------
    ticker : str
        종목 코드 또는 티커 심볼 (예: '005930' 삼성전자, 'AAPL' 애플)
    period : str
        기간 설정 ('1w':1주일, '1m':1개월, '3m':3개월, '6m':6개월, '1y':1년, '3y':3년, '5y':5년, 'ytd':연초부터)
    
    Returns:
    --------
    dict
        수익률 정보를 담은 딕셔너리:
        - 'ticker': 종목 코드
        - 'period': 입력 기간
        - 'start_date': 시작일
        - 'end_date': 종료일
        - 'return': 해당 기간 수익률 (%)
        - 'volatility': 변동성 (%)
        - 'max_drawdown': 최대 낙폭 (%)
        - 'sharpe_ratio': 샤프 비율
    """
    # 종료일은 오늘
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 시작일 계산
    today = datetime.now()
    if period == '1w':
        start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    elif period == '1m':
        start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    elif period == '3m':
        start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
    elif period == '6m':
        start_date = (today - timedelta(days=180)).strftime('%Y-%m-%d')
    elif period == '1y':
        start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    elif period == '3y':
        start_date = (today - timedelta(days=365*3)).strftime('%Y-%m-%d')
    elif period == '5y':
        start_date = (today - timedelta(days=365*5)).strftime('%Y-%m-%d')
    elif period == 'ytd':
        start_date = f"{today.year}-01-01"
    else:
        return {"error": f"유효하지 않은 기간입니다: {period}"}
    
    try:
        # FinanceDataReader로 데이터 가져오기
        df = fdr.DataReader(ticker, start_date, end_date)
        
        if df.empty:
            return {"error": f"데이터를 찾을 수 없습니다: {ticker}"}
        
        # 실제 데이터의 시작일과 종료일
        actual_start_date = df.index[0].strftime('%Y-%m-%d')
        actual_end_date = df.index[-1].strftime('%Y-%m-%d')
        
        # 수익률 계산
        initial_price = df['Close'].iloc[0]
        final_price = df['Close'].iloc[-1]
        total_return = (final_price / initial_price - 1) * 100
        
        # 일별 수익률
        df['daily_return'] = df['Close'].pct_change() * 100
        
        # 변동성 (연율화)
        volatility = df['daily_return'].std() * np.sqrt(252)
        
        # 최대 낙폭
        df['peak'] = df['Close'].cummax()
        df['drawdown'] = (df['Close'] / df['peak'] - 1) * 100
        max_drawdown = df['drawdown'].min()
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        sharpe_ratio = (df['daily_return'].mean() * 252) / (df['daily_return'].std() * np.sqrt(252))
        
        # 결과 딕셔너리
        result = {
            'ticker': ticker,
            'period': period,
            'start_date': actual_start_date,
            'end_date': actual_end_date,
            'return': round(total_return, 2),
            'volatility': round(volatility, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }
        
        return result
    
    except Exception as e:
        return {"error": f"오류 발생: {str(e)}"}



# 함수 사용 예시
if __name__ == "__main__":
    # # 삼성전자 최근 1년 수익률
    # samsung = get_stock_return('005930', '1y')
    # if 'error' not in samsung:
    #     print(f"삼성전자({samsung['ticker']}) {samsung['period']} 수익률:")
    #     print(f"기간: {samsung['start_date']} ~ {samsung['end_date']}")
    #     print(f"수익률: {samsung['return']}%")
    #     print(f"변동성: {samsung['volatility']}%")
    #     print(f"최대낙폭: {samsung['max_drawdown']}%")
    #     print(f"샤프비율: {samsung['sharpe_ratio']}")
    # else:
    #     print(samsung['error'])
    
    # # 애플 최근 3개월 수익률
    # apple = get_stock_return('AAPL', '3m')
    # if 'error' not in apple:
    #     print(f"\n애플({apple['ticker']}) {apple['period']} 수익률:")
    #     print(f"기간: {apple['start_date']} ~ {apple['end_date']}")
    #     print(f"수익률: {apple['return']}%")
    #     print(f"변동성: {apple['volatility']}%")
    #     print(f"최대낙폭: {apple['max_drawdown']}%")
    #     print(f"샤프비율: {apple['sharpe_ratio']}")
    # else:
    #     print(apple['error'])

