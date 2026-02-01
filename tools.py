import yfinance as yf
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class StockTickerInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'RELIANCE.NS', 'HDFCBANK.NS')")

class ValueScoreInput(BaseModel):
    pe_ratio: float = Field(..., description="Price to Earnings ratio. Use -1 if N/A.")
    roe: float = Field(..., description="Return on Equity as decimal (e.g., 0.15 for 15%). Use -1 if N/A.")
    debt_equity: float = Field(..., description="Debt to Equity ratio. Use -1 if N/A.")
    sector: str = Field(..., description="Company sector (e.g., 'Financial Services', 'Industrials', 'Technology')")

# ============================================================================
# SECTOR-SPECIFIC DEBT/EQUITY BENCHMARKS
# These sectors naturally carry high debt as part of their business model
# ============================================================================

SECTOR_DEBT_BENCHMARKS = {
    "Financial Services": {"normal_max": 500, "risky_above": 1000},
    "Banking":            {"normal_max": 500, "risky_above": 1000},
    "Industrials":        {"normal_max": 150, "risky_above": 300},
    "Utilities":          {"normal_max": 200, "risky_above": 400},
    "Real Estate":        {"normal_max": 200, "risky_above": 400},
    "Energy":             {"normal_max": 100, "risky_above": 200},
    "Technology":         {"normal_max": 50,  "risky_above": 100},
    "Consumer Cyclical":  {"normal_max": 80,  "risky_above": 150},
    "Healthcare":         {"normal_max": 60,  "risky_above": 120},
    "Consumer Defensive": {"normal_max": 60,  "risky_above": 120},
    "Basic Materials":    {"normal_max": 80,  "risky_above": 150},
    "Communication Services": {"normal_max": 100, "risky_above": 200},
}

# Default for unknown sectors
DEFAULT_DEBT_BENCHMARK = {"normal_max": 80, "risky_above": 150}

# ============================================================================
# TOOL 1: GET STOCK PRICE AND RSI
# ============================================================================

class GetStockPriceTool(BaseTool):
    name: str = "Get Stock Price and RSI"
    description: str = (
        "Fetches the current stock price and calculates the 14-day RSI "
        "(Relative Strength Index). Use this when you need technical/price data."
    )
    args_schema: Type[BaseModel] = StockTickerInput

    def _run(self, ticker: str) -> str:
        try:
            if not ticker.endswith(('.NS', '.BO')):
                ticker += ".NS"

            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if hist.empty:
                return f"Error: No data found for {ticker}. Check the ticker symbol."

            current_price = hist['Close'].iloc[-1]

            # Calculate 14-day RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            # Interpret RSI
            if rsi < 30:
                rsi_signal = "OVERSOLD - Strong potential BUY signal"
            elif rsi < 45:
                rsi_signal = "Slightly oversold - Good entry point"
            elif rsi < 55:
                rsi_signal = "NEUTRAL - No strong signal"
            elif rsi < 70:
                rsi_signal = "Slightly overbought - Caution"
            else:
                rsi_signal = "OVERBOUGHT - Potential SELL signal"

            return (
                f"Ticker: {ticker} | "
                f"Price: ₹{current_price:.2f} | "
                f"RSI: {rsi:.2f} | "
                f"RSI Signal: {rsi_signal}"
            )
        except Exception as e:
            return f"Error fetching price data for {ticker}: {str(e)}"

# ============================================================================
# TOOL 2: GET STOCK FUNDAMENTALS
# ============================================================================

class GetFundamentalsTool(BaseTool):
    name: str = "Get Stock Fundamentals"
    description: str = (
        "Fetches key financial ratios: P/E, ROE, Debt/Equity, EPS, "
        "and Market Cap. Use this when you need fundamental analysis data."
    )
    args_schema: Type[BaseModel] = StockTickerInput

    def _run(self, ticker: str) -> str:
        try:
            if not ticker.endswith(('.NS', '.BO')):
                ticker += ".NS"

            stock = yf.Ticker(ticker)
            info = stock.info

            sector = info.get('sector', 'Unknown')
            pe = info.get('trailingPE', None)
            roe = info.get('returnOnEquity', None)
            debt_equity = info.get('debtToEquity', None)
            eps = info.get('trailingEps', None)
            market_cap = info.get('marketCap', None)

            # Format each value — clearly mark N/A instead of silently skipping
            pe_str = f"{pe:.2f}" if pe else "N/A"
            roe_str = f"{roe:.2%}" if roe else "N/A"
            de_str = f"{debt_equity:.2f}" if debt_equity else "N/A"
            eps_str = f"₹{eps:.2f}" if eps else "N/A"
            mcap_str = f"₹{market_cap/10000000:.2f} Cr" if market_cap else "N/A"

            # Build a warning list for missing critical data
            missing = []
            if not pe:    missing.append("P/E")
            if not roe:   missing.append("ROE")
            if not debt_equity: missing.append("Debt/Equity")

            missing_warning = ""
            if missing:
                missing_warning = f" | ⚠️ MISSING DATA: {', '.join(missing)} - scoring will be incomplete"

            return (
                f"Ticker: {ticker} | "
                f"Sector: {sector} | "
                f"P/E: {pe_str} | "
                f"ROE: {roe_str} | "
                f"Debt/Equity: {de_str} | "
                f"EPS: {eps_str} | "
                f"Market Cap: {mcap_str}"
                f"{missing_warning}"
            )
        except Exception as e:
            return f"Error fetching fundamentals for {ticker}: {str(e)}"

# ============================================================================
# TOOL 3: CALCULATE VALUE SCORE (SECTOR-AWARE)
# ============================================================================

class CalculateValueScoreTool(BaseTool):
    name: str = "Calculate Value Score"
    description: str = (
        "Calculates a value investment score from 0-100 using SECTOR-AWARE benchmarks. "
        "Pass -1 for any metric that is N/A. The tool will flag missing data clearly "
        "instead of assuming values."
    )
    args_schema: Type[BaseModel] = ValueScoreInput

    def _run(self, pe_ratio: float, roe: float, debt_equity: float, sector: str) -> str:
        score = 0
        max_possible = 0
        details = []
        warnings = []

        # Get sector-specific debt benchmark
        benchmark = SECTOR_DEBT_BENCHMARKS.get(sector, DEFAULT_DEBT_BENCHMARK)

        # --- P/E SCORING (max 35 points) ---
        if pe_ratio == -1 or pe_ratio is None:
            warnings.append("P/E is N/A — P/E score skipped (0/35)")
        else:
            max_possible += 35
            if pe_ratio < 15:
                score += 35
                details.append("Excellent P/E (undervalued)")
            elif pe_ratio < 20:
                score += 25
                details.append("Good P/E")
            elif pe_ratio < 30:
                score += 10
                details.append("Fair P/E")
            else:
                details.append("High P/E — overvaluation risk")

        # --- ROE SCORING (max 35 points) ---
        if roe == -1 or roe is None:
            warnings.append("ROE is N/A — ROE score skipped (0/35)")
        else:
            max_possible += 35
            if roe > 0.20:
                score += 35
                details.append("Excellent ROE — strong management")
            elif roe > 0.15:
                score += 25
                details.append("Good ROE")
            elif roe > 0.10:
                score += 10
                details.append("Fair ROE")
            else:
                details.append("Low ROE — efficiency concerns")

        # --- DEBT/EQUITY SCORING (max 30 points, SECTOR-AWARE) ---
        if debt_equity == -1 or debt_equity is None:
            warnings.append("Debt/Equity is N/A — D/E score skipped (0/30)")
        else:
            max_possible += 30
            normal_max = benchmark["normal_max"]
            risky_above = benchmark["risky_above"]

            if debt_equity < normal_max * 0.3:
                score += 30
                details.append(f"Low debt for {sector} — very safe")
            elif debt_equity < normal_max:
                score += 20
                details.append(f"Normal debt for {sector} — acceptable")
            elif debt_equity < risky_above:
                score += 10
                details.append(f"Elevated debt for {sector} — monitor")
            else:
                details.append(f"High debt even for {sector} — risky (threshold: {risky_above})")

        # --- NORMALIZE SCORE ---
        # If data is missing, normalize score to what's available
        if max_possible > 0:
            normalized_score = int((score / max_possible) * 100)
        else:
            normalized_score = 0

        # Build output
        warning_str = " | ⚠️ " + "; ".join(warnings) if warnings else ""
        detail_str = " | ".join(details) if details else "No data available"

        return (
            f"Sector: {sector} | "
            f"Raw Score: {score}/{max_possible} | "
            f"Normalized Score: {normalized_score}/100 | "
            f"Analysis: {detail_str}"
            f"{warning_str}"
        )

# ============================================================================
# TOOL INSTANCES
# ============================================================================

get_stock_price = GetStockPriceTool()
get_fundamentals = GetFundamentalsTool()
calculate_value_score = CalculateValueScoreTool()