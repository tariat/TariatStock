import os
import sys
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Union
import traceback
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, barssince
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from data_stock_price import get_5m_by_date
from bs4 import BeautifulSoup

class LLMTradingStrategy(Strategy):
    """
    LLM을 활용한 주식 거래 전략
    48개의 과거 데이터를 기반으로 LLM이 매수/매도 판단
    """
    
    def init(self):
        """전략 초기화"""
        # LLM 체인 설정
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
        # 프롬프트 템플릿 설정
        self.prompt_template = PromptTemplate(
            input_variables=["market_data", "current_price", "position_status"],
            template="""
            당신은 전문적인 주식 트레이더입니다. 아래 데이터를 분석하여 투자 결정을 내려주세요.

            최근 48개 5분봉 데이터:
            {market_data}

            현재 가격: {current_price}
            현재 포지션: {position_status}

            분석할 요소:
            1. 가격 추세 (상승/하락/횡보)
            2. 거래량 변화
            3. RSI, 이동평균 등 기술적 지표
            4. 지지선/저항선
            5. 현재 포지션 고려

            답변을 'format'과 같이 해 주세요.
            - reason: 투자 결정을 하게 된 근거
            - descision: BUY, SELL, HOLD
            
            'format'            
            <reason>reason</reason>
            <descision>decision</descision>
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        # 거래 기록
        self.trade_decisions = []
    
    def get_market_data_text(self, lookback=48):
        """최근 48개 데이터를 텍스트 형태로 변환"""
        current_idx = len(self.data) - 1
        start_idx = max(0, current_idx - lookback + 1)
        
        market_data = []
        for i in range(start_idx, current_idx + 1):
            data_point = {
                'time': i,
                'open': self.data.Open[i],
                'high': self.data.High[i],
                'low': self.data.Low[i],
                'close': self.data.Close[i],
                'volume': getattr(self.data, 'Volume', [0])[i] if hasattr(self.data, 'Volume') else 0
            }
            market_data.append(
                f"시점 {i}: 시가={data_point['open']:.0f}, "
                f"고가={data_point['high']:.0f}, 저가={data_point['low']:.0f}, "
                f"종가={data_point['close']:.0f}, 거래량={data_point['volume']:.0f}"
            )
        
        return "\n".join(market_data)
    
    def get_llm_decision(self):
        """LLM을 통한 거래 결정"""
        try:
            # 시장 데이터 준비
            market_data_text = self.get_market_data_text()
            current_price = self.data.Close[-1]
            position_status = "보유 중" if self.position else "미보유"
            
            # LLM 호출
            response = self.chain.run(
                market_data=market_data_text,
                current_price=current_price,
                position_status=position_status
            )
            
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response, "html.parser")

            decision_tag = soup.find("descision")
            reason_tag = soup.find("reason")

            decision = decision_tag.get_text(strip=True).upper() if decision_tag else "HOLD"
            reason = reason_tag.get_text(strip=True) if reason_tag else "이유 없음"

            print(f"LLM 결정: {decision} | 근거: {reason}")

            return decision, reason
                
        except Exception as e:
            print(f"LLM 결정 오류: {e}")
            return "HOLD", "'"
    
    def next(self):
        """각 시점에서 실행되는 거래 로직"""
        # 충분한 데이터가 없으면 대기
        if len(self.data) < 48:
            return
        
        current_price = self.data.Close[-1]
        
        # LLM 결정 획득
        decision, reason = self.get_llm_decision()
        
        # 거래 결정 기록
        self.trade_decisions.append({
            'timestamp': len(self.data) - 1,
            'price': current_price,
            'decision': decision,
            'decision': reason,
            'position': bool(self.position)
        })
        
        # 거래 실행
        if decision == "BUY" and not self.position:
            # 매수: 손절매를 5% 아래로 설정
            self.buy(sl=0.95 * current_price)
            print(f"매수 실행: {current_price:.0f} (손절매: {0.95 * current_price:.0f})")
            
        elif decision == "SELL" and self.position:
            # 매도
            self.position.close()
            print(f"매도 실행: {current_price:.0f}")

def load_stock_data(stock_code: str, date: str) -> pd.DataFrame:
    """
    주식 5분봉 데이터 로드 및 백테스팅 형식으로 변환
    
    Args:
        stock_code: 종목코드 (예: "009540")
        date: 거래일 (예: "20250515")
    
    Returns:
        백테스팅에 사용할 DataFrame
    """
    try:
        current_date = datetime.strptime(date, "%Y%m%d")
        previous_date = current_date - timedelta(days=1)
        previous_date_str = previous_date.strftime("%Y%m%d")

        # 5분봉 데이터 가져오기
        data = get_5m_by_date(stock_code, date)
        data2 = get_5m_by_date(stock_code, previous_date_str)

        tot = pd.concat([data,data2], axis=0)
        tot = tot.sort_index()

        tot['data'] = [tot.iloc[max(0, i - 47):i + 1].to_dict('list') for i in range(len(tot))]
        
        # 컬럼명 표준화
        if 'datetime' in tot.columns:
            tot = tot.rename({'datetime': 'Date'}, axis=1)
        elif 'timestamp' in data.columns:
            tot = tot.rename({'timestamp': 'Date'}, axis=1)
                
        # 필요한 컬럼 확인 및 변환
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trading_value', 'data']
        for col in required_columns:
            if col.lower() in data.columns:
                data = data.rename({col.lower(): col}, axis=1)
        
        # 인덱스 설정
        if 'Date' in data.columns:
            data = data.set_index('Date')
        
        # 데이터 타입 변환
        for col in required_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 결측치 제거
        data = data.dropna()
        
        # 최소 필요 데이터 확인
        if len(data) < 48:
            raise ValueError(f"데이터가 부족합니다. 필요: 48개, 실제: {len(data)}개")
        
        print(f"데이터 로드 완료: {len(data)}개 레코드")
        print(f"데이터 기간: {data.index[0]} ~ {data.index[-1]}")
        
        return data
        
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        raise

def run_backtest(stock_code: str, date: str, initial_cash: int = 10000000):
    """
    백테스팅 실행
    
    Args:
        stock_code: 종목코드
        date: 거래일
        initial_cash: 초기 자금 (기본: 1천만원)
    """
    try:
        print(f"\n=== 백테스팅 시작 ===")
        print(f"종목: {stock_code}")
        print(f"날짜: {date}")
        print(f"초기 자금: {initial_cash:,}원")
        
        # 데이터 로드
        data = load_stock_data(stock_code, date)
        
        # 백테스팅 실행
        bt = Backtest(
            data=data,
            strategy=LLMTradingStrategy,
            commission=0.002,  # 수수료 0.2%
            cash=initial_cash,
            exclusive_orders=True
        )
        
        # 백테스팅 실행
        print("\n백테스팅 실행 중...")
        results = bt.run()
        
        # 결과 출력
        print(f"\n=== 백테스팅 결과 ===")
        print(f"최종 수익률: {results['Return [%]']:.2f}%")
        print(f"최종 자산: {results['Equity Final [$]']:,.0f}원")
        print(f"최대 손실률: {results['Max. Drawdown [%]']:.2f}%")
        print(f"거래 횟수: {results['# Trades']}")
        print(f"승률: {results['Win Rate [%]']:.2f}%")
        print(f"평균 수익률: {results['Avg. Trade [%]']:.2f}%")
        print(f"샤프 비율: {results['Sharpe Ratio']:.2f}")
        
        # 상세 결과 반환
        return {
            'results': results,
            'backtest': bt,
            'data': data
        }
        
    except Exception as e:
        print(f"백테스팅 오류: {e}")
        traceback.print_exc()
        return None

def analyze_trades(backtest_result):
    """거래 내역 분석"""
    if not backtest_result:
        return
    
    bt = backtest_result['backtest']
    
    # 거래 내역이 있는 경우 분석
    if hasattr(bt._strategy, 'trade_decisions'):
        decisions = bt._strategy.trade_decisions
        
        print(f"\n=== 거래 결정 분석 ===")
        print(f"총 결정 횟수: {len(decisions)}")
        
        buy_decisions = [d for d in decisions if d['decision'] == 'BUY']
        sell_decisions = [d for d in decisions if d['decision'] == 'SELL']
        hold_decisions = [d for d in decisions if d['decision'] == 'HOLD']
        
        print(f"매수 신호: {len(buy_decisions)}회")
        print(f"매도 신호: {len(sell_decisions)}회")
        print(f"관망 신호: {len(hold_decisions)}회")

# 메인 실행부
if __name__ == "__main__":
    # 백테스팅 실행
    result = run_backtest("009540", "20250516")
    
    if result:
        # 거래 분석
        analyze_trades(result)
        
        # 결과 시각화 (선택사항)
        try:
            result['backtest'].plot()
        except Exception as e:
            print(f"차트 출력 오류: {e}")
    
    print("\n백테스팅 완료!")