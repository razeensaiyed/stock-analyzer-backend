import os
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import quantitative_analyst, research_analyst, portfolio_manager
from tasks import create_quant_task, create_research_task, create_decision_task
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

# ============================================================================
# OUTPUT SAVING
# ============================================================================

def save_result(ticker: str, result: str):
    """Save each analysis to both a log file and a CSV summary."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Save full report to text log
    log_filename = "analysis_log.txt"
    with open(log_filename, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ANALYSIS: {ticker} | Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n")
        f.write(str(result) + "\n\n")

    # 2. Parse decision from result for CSV summary
    result_str = str(result).upper()
    decision = "UNKNOWN"
    confidence = "UNKNOWN"
    risk = "UNKNOWN"

    for line in str(result).split("\n"):
        line_upper = line.upper().strip()
        if "FINAL DECISION:" in line_upper:
            for d in ["BUY", "SELL", "WATCH", "AVOID"]:
                if d in line_upper:
                    decision = d
                    break
        if "CONFIDENCE:" in line_upper:
            for c in ["HIGH", "MEDIUM", "LOW"]:
                if c in line_upper:
                    confidence = c
                    break
        if "RISK LEVEL:" in line_upper:
            for r in ["HIGH", "MEDIUM", "LOW"]:
                if r in line_upper:
                    risk = r
                    break

    # BUG 1 FIX: Deterministic confidence downgrade if missing data detected
    result_text = str(result)
    if "⚠️" in result_text or "MISSING DATA" in result_text.upper():
        # Downgrade confidence one level
        if confidence == "HIGH":
            confidence = "MEDIUM"
            print(f"   ⚠️  Confidence downgraded HIGH → MEDIUM (missing data detected)")
        elif confidence == "MEDIUM":
            confidence = "LOW"
            print(f"   ⚠️  Confidence downgraded MEDIUM → LOW (missing data detected)")

    # 3. Append to CSV summary
    csv_filename = "analysis_summary.csv"
    file_exists = os.path.exists(csv_filename)

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Decision", "Confidence", "Risk Level"])
        writer.writerow([timestamp, ticker, decision, confidence, risk])

    print(f"\n💾 Results saved:")
    print(f"   Full report → {log_filename}")
    print(f"   Summary     → {csv_filename}")

    # Return parsed data for JSON export
    return {
        'timestamp': timestamp,
        'ticker': ticker,
        'decision': decision,
        'confidence': confidence,
        'risk': risk,
        'full_analysis': str(result)
    }


# ============================================================================
# CORE ANALYSIS FUNCTION
# ============================================================================

def analyze_stock(ticker: str) -> dict:
    """Run complete multi-agent analysis on one stock."""

    print("=" * 80)
    print(f"🤖 MULTI-AGENT STOCK ANALYSIS: {ticker}")
    print("=" * 80)
    print("\nInitializing agents and tasks...")

    quant_task    = create_quant_task(ticker)
    research_task = create_research_task(ticker)
    decision_task = create_decision_task(ticker)

    crew = Crew(
        agents=[quantitative_analyst, research_analyst, portfolio_manager],
        tasks=[quant_task, research_task, decision_task],
        process=Process.sequential,
        verbose=True,
        cache=False  # Disable caching so tools always fetch fresh data
    )

    print("\n🚀 Starting analysis...\n")
    result = crew.kickoff()

    print("\n" + "=" * 80)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n" + str(result))
    print("\n" + "=" * 80)

    # Save to files and get parsed data
    parsed_data = save_result(ticker, result)

    return {
        'ticker': ticker,
        'result': str(result),
        'status': 'completed',
        'parsed_data': parsed_data
    }


# PHASE 3: PARALLEL EXECUTION
def analyze_multiple_stocks(tickers: list, parallel: bool = True, max_workers: int = 3) -> list:
    """
    Analyze multiple stocks.
    
    Args:
        tickers: List of stock tickers
        parallel: If True, run stocks in parallel (much faster)
        max_workers: Number of parallel workers (default 3 to avoid rate limits)
    """
    results = []

    if not parallel:
        # Sequential execution (original behavior)
        for i, ticker in enumerate(tickers, 1):
            print(f"\n\n{'=' * 80}")
            print(f"ANALYZING STOCK {i}/{len(tickers)}")
            print(f"{'=' * 80}\n")

            try:
                result = analyze_stock(ticker)
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error analyzing {ticker}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'result': f'Error: {str(e)}',
                    'status': 'failed',
                    'parsed_data': None
                })
    else:
        # PARALLEL EXECUTION - 4-5x faster for multiple stocks
        print(f"\n🚀 PARALLEL MODE: Running {len(tickers)} stocks with {max_workers} workers\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all stocks for parallel processing
            future_to_ticker = {
                executor.submit(analyze_stock, ticker): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    print(f"\n✅ [{completed}/{len(tickers)}] {ticker} completed")
                except Exception as e:
                    print(f"\n❌ [{completed}/{len(tickers)}] {ticker} failed: {str(e)}")
                    results.append({
                        'ticker': ticker,
                        'result': f'Error: {str(e)}',
                        'status': 'failed',
                        'parsed_data': None
                    })

    return results


def save_results_to_json(results: list, filename: str = "results.json"):
    """Export analysis results to JSON for frontend consumption."""
    json_data = []
    
    for r in results:
        if r['status'] == 'completed' and r.get('parsed_data'):
            parsed = r['parsed_data']
            
            # Extract key reasons and action from full analysis
            full_text = parsed['full_analysis']
            key_reasons = []
            specific_action = ""
            
            # Simple parsing - can be improved
            lines = full_text.split('\n')
            in_reasons = False
            in_action = False
            
            for line in lines:
                if 'Key Reasons:' in line or 'key reasons' in line.lower():
                    in_reasons = True
                    continue
                if 'Specific Action:' in line or 'specific action' in line.lower():
                    in_reasons = False
                    in_action = True
                    continue
                
                if in_reasons and line.strip() and line.strip()[0].isdigit():
                    key_reasons.append(line.strip())
                elif in_action and line.strip():
                    specific_action = line.strip()
                    break
            
            json_data.append({
                'ticker': parsed['ticker'],
                'timestamp': parsed['timestamp'],
                'decision': parsed['decision'],
                'confidence': parsed['confidence'],
                'risk_level': parsed['risk'],
                'key_reasons': key_reasons,
                'specific_action': specific_action,
                'full_analysis': parsed['full_analysis']
            })
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n📄 JSON export saved to {filename}")
    return filename


def load_tickers_from_csv(csv_path: str) -> list:
    """Load tickers from a CSV file. Expects a column named 'Ticker' or 'Symbol'."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    if 'Ticker' in df.columns:
        tickers = df['Ticker'].tolist()
    elif 'Symbol' in df.columns:
        tickers = df['Symbol'].tolist()
    else:
        # Use first column
        tickers = df.iloc[:, 0].tolist()

    # Auto-add .NS if missing
    tickers = [str(t).strip() for t in tickers]
    tickers = [t if t.endswith(('.NS', '.BO')) else f"{t}.NS" for t in tickers]

    return tickers


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🎯 DAD'S STOCK ANALYSIS SYSTEM\n")
    print("Choose analysis mode:")
    print("1. Single stock")
    print("2. Multiple stocks (type them in)")
    print("3. Load from CSV file")
    print("4. Quick test (HDFCBANK.NS)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        ticker = input("Enter stock ticker (e.g., RELIANCE.NS): ").strip().upper()
        if not ticker.endswith(('.NS', '.BO')):
            ticker += ".NS"
        result = analyze_stock(ticker)
        
        # Save to JSON
        save_results_to_json([result])

    elif choice == "2":
        tickers_input = input("Enter tickers separated by commas (e.g., HDFCBANK.NS,RELIANCE.NS): ")
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        tickers = [t if t.endswith(('.NS', '.BO')) else f"{t}.NS" for t in tickers]

        # Ask if user wants parallel execution
        use_parallel = input("\nUse parallel execution for faster analysis? (y/n, default=y): ").strip().lower()
        parallel = use_parallel != 'n'
        
        results = analyze_multiple_stocks(tickers, parallel=parallel)

        print("\n\n" + "=" * 80)
        print("📈 BATCH ANALYSIS SUMMARY")
        print("=" * 80)
        for r in results:
            emoji = "✅" if r['status'] == 'completed' else "❌"
            print(f"{emoji} {r['ticker']}: {r['status']}")
        print(f"\n💾 All results saved to analysis_log.txt and analysis_summary.csv")
        
        # Save to JSON
        save_results_to_json(results)

    elif choice == "3":
        csv_path = input("Enter path to CSV file: ").strip()
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
        else:
            tickers = load_tickers_from_csv(csv_path)
            print(f"\n📄 Loaded {len(tickers)} tickers from {csv_path}")
            print(f"   Tickers: {', '.join(tickers)}")
            
            # Ask if user wants parallel execution
            use_parallel = input(f"\nUse parallel execution for faster analysis? (y/n, default=y): ").strip().lower()
            parallel = use_parallel != 'n'
            
            confirm = input(f"\nAnalyze all {len(tickers)} stocks? (y/n): ").strip().lower()
            if confirm == 'y':
                results = analyze_multiple_stocks(tickers, parallel=parallel)
                print(f"\n💾 All results saved to analysis_log.txt and analysis_summary.csv")
                
                # Save to JSON
                save_results_to_json(results)

    elif choice == "4":
        print("\n🧪 Running quick test with HDFCBANK.NS...\n")
        result = analyze_stock("HDFCBANK.NS")
        
        # Save to JSON
        save_results_to_json([result])

    else:
        print("Invalid choice!")

    print("\n✅ Done!")