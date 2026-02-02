from crewai import Task
from agents import quantitative_analyst, research_analyst, portfolio_manager
from tools_hybrid import get_stock_price, get_fundamentals, calculate_value_score
from news_tools import get_google_news

# ============================================================================
# TASK 1: QUANTITATIVE ANALYSIS
# ============================================================================

def create_quant_task(ticker: str) -> Task:
    return Task(
        description=f"""Perform comprehensive quantitative analysis on {ticker}.

        Step-by-step requirements:
        1. Use 'Get Stock Price and RSI' tool to fetch price and RSI
        2. Use 'Get Stock Fundamentals' tool to get P/E, ROE, Debt/Equity, and Sector
        3. Extract the numerical values carefully:
           - If any value is N/A, pass -1 for that value to the scoring tool
           - Do NOT make up or assume values. Pass -1 exactly as instructed.
        4. Use 'Calculate Value Score' tool. You MUST pass the sector string exactly 
           as returned by the fundamentals tool. The tool is sector-aware.
        5. Analyze the RSI signal returned by the price tool
        6. Apply decision rules using the NORMALIZED score (not raw score):
           - BUY: Normalized score > 60 AND RSI < 45
           - WATCH: Normalized score 40-60 OR RSI 45-70
           - AVOID: Normalized score < 40 OR RSI > 70
        7. If critical data is missing (flagged by ⚠️), state this clearly 
           and factor it into your confidence level

        Your output MUST include:
        - Ticker and sector
        - All metrics exactly as returned (do not round or modify)
        - Which metrics are missing and why that affects confidence
        - Normalized Value Score
        - RSI and its signal
        - Your recommendation: BUY, WATCH, or AVOID
        - Your confidence: HIGH, MEDIUM, or LOW
        - Reasoning (2-3 sentences)""",
        expected_output="""Structured quantitative analysis with all metrics, 
        normalized score, missing data flags, recommendation, and confidence level.""",
        agent=quantitative_analyst,
        tools=[get_stock_price, get_fundamentals, calculate_value_score]
    )

# ============================================================================
# TASK 2: QUALITATIVE RESEARCH (NOW WITH NEWS)
# ============================================================================

def create_research_task(ticker: str) -> Task:
    return Task(
        description=f"""Conduct qualitative research on {ticker}.

        IMPORTANT: You must be transparent about what you actually know vs what 
        you are inferring. Do NOT present guesses as facts.

        Steps:
        1. Use 'Get Stock Fundamentals' to get the sector
        2. Use 'Get Google News' to get recent news about the company
        3. Based on the sector and news, provide qualitative analysis BUT clearly label 
           each point as one of:
           - [KNOWN] - Well-established facts about the company or from recent news
           - [INFERRED] - Reasonable inference based on sector/known data
           - [NEEDS VERIFICATION] - Something that should be checked against 
             current news/annual reports before acting on

        4. Analyze:
           - Competitive position in sector
           - Business model strength
           - Key risks
           - Growth prospects
           - Recent news sentiment (positive/negative/neutral)

        5. Give sentiment: BULLISH, NEUTRAL, or BEARISH
        6. Give confidence: HIGH, MEDIUM, or LOW
           - If most points are [INFERRED] or [NEEDS VERIFICATION], 
             confidence MUST be LOW or MEDIUM

        Your output MUST include:
        - Sector and business overview
        - News sentiment summary (if news available)
        - 2-3 strengths (each labeled [KNOWN]/[INFERRED]/[NEEDS VERIFICATION])
        - 2-3 risks (each labeled)
        - Sector outlook (labeled)
        - Sentiment and confidence
        - A clear note on what data would improve this analysis""",
        expected_output="""Qualitative analysis with transparently labeled points, 
        news sentiment, confidence level, and data gaps identified.""",
        agent=research_analyst,
        tools=[get_fundamentals, get_google_news]  # Added news tool
    )

# ============================================================================
# TASK 3: FINAL DECISION
# ============================================================================

def create_decision_task(ticker: str) -> Task:
    return Task(
        description=f"""Make the FINAL investment decision on {ticker}.

        You will receive outputs from Agent 1 (Quantitative) and Agent 2 (Research).

        STRICT RULES - You MUST follow these:

        1. DECISION MATRIX (follow exactly, do not override):
           - Quant BUY  + Research BULLISH  = BUY   | Confidence: HIGH
           - Quant BUY  + Research NEUTRAL  = BUY   | Confidence: MEDIUM
           - Quant BUY  + Research BEARISH  = WATCH | Confidence: MEDIUM
           - Quant WATCH + Research BULLISH = BUY   | Confidence: MEDIUM
           - Quant WATCH + Research NEUTRAL = WATCH | Confidence: MEDIUM
           - Quant WATCH + Research BEARISH = WATCH | Confidence: LOW
           - Quant AVOID + Research BULLISH = WATCH | Confidence: LOW
           - Quant AVOID + Research NEUTRAL = AVOID | Confidence: MEDIUM
           - Quant AVOID + Research BEARISH = AVOID | Confidence: HIGH

        2. CONFIDENCE ADJUSTMENT:
           - Note: The system will auto-downgrade confidence if missing data is detected.
           - You should mention if data is missing but do NOT manually downgrade.

        3. RISK LEVEL - CRITICAL RULES:
           - The Quantitative Analyst already assessed Debt/Equity using SECTOR-AWARE 
             benchmarks. If the tool output said "Normal" or "Acceptable" for debt, 
             do NOT override that to HIGH risk based on the raw number alone.
           - Risk = HIGH only if: tool explicitly flagged debt as "risky" or "high"
           - Risk = MEDIUM if: any metric is N/A
           - Otherwise use your judgment based on the overall assessment

        Your output MUST include:
        - Summary of Agent 1 findings (1-2 sentences)
        - Summary of Agent 2 findings (1-2 sentences, include news sentiment if available)
        - Which row of the decision matrix you applied and why
        - FINAL DECISION: [BUY/SELL/WATCH/AVOID]
        - CONFIDENCE: [HIGH/MEDIUM/LOW]
        - RISK LEVEL: [LOW/MEDIUM/HIGH]
        - 3 key reasons
        - Specific action (e.g. "Buy 5% position", "Wait for RSI < 30")""",
        expected_output="""Final decision with explicit decision matrix row applied, 
        confidence adjustments noted, and actionable recommendation.""",
        agent=portfolio_manager,
        tools=[]
    )