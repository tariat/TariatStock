import pandas as pd
from data_stock_price import get_5m_by_date

from typing import Dict, List, Optional
import json

from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Union
import traceback
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, barssince
from backtesting.test import SMA, RSI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from data_stock_price import get_5m_by_date

class EnhancedLLMTradingStrategy(Strategy):
    """
    개선된 LLM 거래 전략 - 고급 기능 포함
    """

    def init(self):
        super().init()
        
        # # 고급 기술적 지표
        # self.sma_5 = self.I(SMA, self.data.Close, 5)
        # self.sma_10 = self.I(SMA, self.data.Close, 10)
        # self.sma_20 = self.I(SMA, self.data.Close, 20)
        # self.rsi = self.I(RSI, self.data.Close, 14)
        
        # # 위험 관리 매개변수
        # self.max_risk_per_trade = 0.02  # 거래당 최대 2% 위험
        # self.consecutive_losses = 0
        # self.max_consecutive_losses = 3
        
        # 성과 추적
        self.performance_log = []
        
    def calculate_technical_indicators(self) -> Dict:
        """기술적 지표 계산"""
        current_idx = len(self.data) - 1
        
        return {
            'price': self.data.Close[-1],
            'sma_5': self.sma_5[-1],
            'sma_10': self.sma_10[-1],
            'sma_20': self.sma_20[-1],
            'rsi': self.rsi[-1],
            'price_change_1': (self.data.Close[-1] / self.data.Close[-2] - 1) * 100 if len(self.data) > 1 else 0,
            'price_change_5': (self.data.Close[-1] / self.data.Close[-6] - 1) * 100 if len(self.data) > 5 else 0,
            'volume_avg': np.mean(getattr(self.data, 'Volume', [0] * len(self.data))[-5:]) if hasattr(self.data, 'Volume') else 0
        }
    
    def create_enhanced_prompt(self) -> str:
        """향상된 프롬프트 생성"""
        indicators = self.calculate_technical_indicators()
        market_data_text = self.get_market_data_text()
        
        return f"""
        당신은 전문 퀀트 트레이더입니다. 다음 정보를 종합적으로 분석하여 투자 결정을 내려주세요.

        === 현재 시장 상황 ===
        현재가: {indicators['price']:.0f}원
        RSI: {indicators['rsi']:.1f}
        5일 이평: {indicators['sma_5']:.0f}원
        10일 이평: {indicators['sma_10']:.0f}원  
        20일 이평: {indicators['sma_20']:.0f}원
        1봉 변화율: {indicators['price_change_1']:.2f}%
        5봉 변화율: {indicators['price_change_5']:.2f}%

        === 최근 48개 5분봉 데이터 ===
        {market_data_text}

        === 현재 포지션 ===
        보유 상태: {"보유 중" if self.position else "미보유"}
        연속 손실: {self.consecutive_losses}회

        === 분석 요청 ===
        1. 추세 분석 (강세/약세/횡보)
        2. 모멘텀 상태 
        3. 지지선/저항선 위치
        4. 리스크 수준 평가
        5. 최적 진입/청산 타이밍

        === 응답 형식 ===
        다음 JSON 형식으로만 응답해주세요:
        {{
            "decision": "BUY/SELL/HOLD",
            "confidence": 0.7,
            "reason": "결정 이유 설명",
            "stop_loss": 0.95,
            "take_profit": 1.05
        }}
        """
    
    def parse_llm_response(self, response: str) -> Dict:
        """LLM 응답 파싱"""
        try:
            # JSON 형태로 파싱 시도
            result = json.loads(response.strip())
            return {
                'decision': result.get('decision', 'HOLD').upper(),
                'confidence': float(result.get('confidence', 0.5)),
                'reason': result.get('reason', ''),
                'stop_loss': float(result.get('stop_loss', 0.95)),
                'take_profit': float(result.get('take_profit', 1.05))
            }
        except:
            # 기본 파싱
            response_upper = response.upper()
            if 'BUY' in response_upper:
                decision = 'BUY'
            elif 'SELL' in response_upper:
                decision = 'SELL'
            else:
                decision = 'HOLD'
            
            return {
                'decision': decision,
                'confidence': 0.5,
                'reason': response,
                'stop_loss': 0.95,
                'take_profit': 1.05
            }
    
    def get_position_size(self, price: float, stop_loss: float) -> float:
        """적정 포지션 사이즈 계산"""
        account_value = self.equity
        risk_amount = account_value * self.max_risk_per_trade
        price_risk = abs(price - stop_loss * price)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
            return min(position_size, account_value * 0.3)  # 최대 30% 투자
        return account_value * 0.1  # 기본 10% 투자
    
    def next(self):
        """개선된 거래 로직"""
        if len(self.data) < 48:
            return
        
        # 연속 손실 제한
        if self.consecutive_losses >= self.max_consecutive_losses:
            return
        
        try:
            # Enhanced prompt 사용
            prompt = self.create_enhanced_prompt()
            response = self.chain.run(prompt)
            parsed_response = self.parse_llm_response(response)
            
            current_price = self.data.Close[-1]
            decision = parsed_response['decision']
            confidence = parsed_response['confidence']
            
            # 신뢰도가 낮으면 거래하지 않음
            if confidence < 0.6:
                decision = 'HOLD'
            
            # 거래 실행
            if decision == "BUY" and not self.position:
                stop_loss_price = current_price * parsed_response['stop_loss']
                position_size = self.get_position_size(current_price, stop_loss_price)
                
                self.buy(
                    size=position_size / current_price,
                    sl=stop_loss_price
                )
                
                self.performance_log.append({
                    'action': 'BUY',
                    'price': current_price,
                    'time': len(self.data),
                    'confidence': confidence,
                    'reason': parsed_response['reason']
                })
                
            elif decision == "SELL" and self.position:
                self.position.close()
                
                # 손익 확인하여 연속 손실 카운팅
                if self.position.pl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                self.performance_log.append({
                    'action': 'SELL',
                    'price': current_price,
                    'time': len(self.data),
                    'confidence': confidence,
                    'reason': parsed_response['reason'],
                    'pl': self.position.pl
                })
                
        except Exception as e:
            print(f"거래 실행 오류: {e}")

def run_enhanced_backtest(stock_code: str, date: str, 
                         llm_provider: str = "openai",
                         model_name: str = "gpt-3.5-turbo"):
    """
    개선된 백테스팅 실행
    
    Args:
        stock_code: 종목코드
        date: 거래일
        llm_provider: LLM 공급업체 (openai, anthropic 등)
        model_name: 모델명
    """
    print(f"\n=== 향상된 백테스팅 시작 ===")
    print(f"LLM: {llm_provider} - {model_name}")
    
    # LLM 설정
    if llm_provider == "openai":
        from langchain.llms import OpenAI
        llm = OpenAI(model_name=model_name, temperature=0.1)
    else:
        raise ValueError(f"지원하지 않는 LLM: {llm_provider}")
    
    # 데이터 로드
    data = load_stock_data(stock_code, date)
    
    # 전략 클래스에 LLM 설정
    EnhancedLLMTradingStrategy.llm = llm
    
    # 백테스팅 실행
    bt = Backtest(
        data=data,
        strategy=EnhancedLLMTradingStrategy,
        commission=0.002,
        cash=10000000,
        exclusive_orders=True
    )
    
    results = bt.run()
    
    # 상세 분석
    analyze_enhanced_results(results, bt)
    
    return {
        'results': results,
        'backtest': bt,
        'performance_log': bt._strategy.performance_log
    }

def analyze_enhanced_results(results, bt):
    """상세 결과 분석"""
    print(f"\n=== 상세 백테스팅 결과 ===")
    print(f"수익률: {results['Return [%]']:.2f}%")
    print(f"연평균 수익률: {results['Return (Ann.) [%]']:.2f}%")
    print(f"변동성: {results['Volatility (Ann.) [%]']:.2f}%")
    print(f"샤프 비율: {results['Sharpe Ratio']:.2f}")
    print(f"최대 손실률: {results['Max. Drawdown [%]']:.2f}%")
    print(f"승률: {results['Win Rate [%]']:.2f}%")
    print(f"수익 팩터: {results['Profit Factor']:.2f}")
    
    # 성과 로그 분석
    if hasattr(bt._strategy, 'performance_log'):
        log = bt._strategy.performance_log
        buy_trades = [t for t in log if t['action'] == 'BUY']
        sell_trades = [t for t in log if t['action'] == 'SELL']
        
        print(f"\n매수 거래: {len(buy_trades)}회")
        print(f"매도 거래: {len(sell_trades)}회")
        
        if sell_trades:
            profits = [t['pl'] for t in sell_trades if 'pl' in t]
            if profits:
                print(f"평균 손익: {np.mean(profits):,.0f}원")
                print(f"최대 이익: {max(profits):,.0f}원")
                print(f"최대 손실: {min(profits):,.0f}원")

if __name__ == "__main__":
    # 향상된 백테스팅 실행
    result = run_enhanced_backtest("009540", "20250515")
    
    if result:
        # 거래 내역 상세 분석
        print("\n=== 거래 내역 상세 ===")
        for trade in result['performance_log']:
            print(f"{trade['action']} @ {trade['price']:,.0f}원 "
                  f"(신뢰도: {trade['confidence']:.2f}) - {trade['reason'][:50]}...")
            
