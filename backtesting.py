# backtesting.py - Add to your project

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from crewai import Crew, Process
from agents import quantitative_analyst, research_analyst, portfolio_manager
from tasks import create_quant_task, create_research_task, create_decision_task

def backtest_strategy(tickers: list, start_date: str, end_date: str, lookback_days: int = 30):
    """
    Run historical analysis to validate decision accuracy.
    Compare recommendations against actual stock performance.
    
    Args:
        tickers: List of stock tickers to backtest
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format  
        lookback_days: Days to measure performance after recommendation (default: 30)
    
    Returns:
        Dictionary with backtest results
    """
    results = []
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Backtesting: {ticker}")
        print(f"{'='*60}")
        
        try:
            # Get historical price at decision date
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"No data for {ticker}")
                continue
            
            # Simulate running analysis at start_date
            # (In real backtest, you'd need historical fundamental data)
            print(f"Running analysis as of {start_date}...")
            
            quant_task = create_quant_task(ticker)
            research_task = create_research_task(ticker)
            decision_task = create_decision_task(ticker)
            
            crew = Crew(
                agents=[quantitative_analyst, research_analyst, portfolio_manager],
                tasks=[quant_task, research_task, decision_task],
                process=Process.sequential,
                verbose=False,
                cache=False
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            # Parse decision
            decision = "UNKNOWN"
            for line in result_str.split("\n"):
                if "FINAL DECISION:" in line.upper():
                    for d in ["BUY", "SELL", "WATCH", "AVOID"]:
                        if d in line.upper():
                            decision = d
                            break
            
            # Calculate actual performance
            start_price = hist['Close'].iloc[0]
            
            # Get price after lookback_days
            future_date = pd.to_datetime(start_date) + timedelta(days=lookback_days)
            future_hist = stock.history(start=future_date, end=future_date + timedelta(days=5))
            
            if not future_hist.empty:
                end_price = future_hist['Close'].iloc[0]
                actual_return = ((end_price - start_price) / start_price) * 100
                
                # Evaluate accuracy
                correct = False
                if decision == "BUY" and actual_return > 0:
                    correct = True
                elif decision == "AVOID" and actual_return < 0:
                    correct = True
                elif decision == "WATCH":
                    correct = abs(actual_return) < 5  # Neutral is correct if <5% change
                
                result_data = {
                    'ticker': ticker,
                    'date': start_date,
                    'decision': decision,
                    'start_price': start_price,
                    'end_price': end_price,
                    'actual_return': actual_return,
                    'correct': correct,
                    'lookback_days': lookback_days
                }
                
                results.append(result_data)
                
                emoji = "✅" if correct else "❌"
                print(f"{emoji} Decision: {decision} | Return: {actual_return:.2f}% | Correct: {correct}")
            else:
                print(f"No future price data available")
        
        except Exception as e:
            print(f"Error backtesting {ticker}: {str(e)}")
    
    # Calculate overall accuracy
    if results:
        accuracy = sum(1 for r in results if r['correct']) / len(results) * 100
        avg_return_on_buys = sum(r['actual_return'] for r in results if r['decision'] == 'BUY') / max(sum(1 for r in results if r['decision'] == 'BUY'), 1)
        
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {len(results)}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Avg return on BUY decisions: {avg_return_on_buys:.2f}%")
        print(f"Lookback period: {lookback_days} days")
        
        return {
            'results': results,
            'accuracy': accuracy,
            'avg_return_on_buys': avg_return_on_buys,
            'total_tests': len(results)
        }
    else:
        return {'results': [], 'accuracy': 0, 'error': 'No successful backtests'}


# Quick backtest function for demo
def quick_backtest_demo():
    """
    Run a quick backtest demo on a few popular stocks.
    """
    print("\n🔬 RUNNING BACKTEST DEMO\n")
    
    # Test on popular stocks from 30 days ago
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    results = backtest_strategy(tickers, start_date, end_date, lookback_days=30)
    
    return results


if __name__ == "__main__":
    # Run demo
    quick_backtest_demo()