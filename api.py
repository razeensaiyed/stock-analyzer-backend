from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import quantitative_analyst, research_analyst, portfolio_manager
from tasks import create_quant_task, create_research_task, create_decision_task
   
load_dotenv()
   
app = Flask(__name__)
CORS(app)  # Allow frontend to call this API
   
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker', '').strip().upper()
       
    if not ticker:
        return jsonify({'error': 'Ticker required'}), 400
       
    if not ticker.endswith(('.NS', '.BO')):
        ticker += '.NS'
       
    try:
           # Run the analysis
        quant_task = create_quant_task(ticker)
        research_task = create_research_task(ticker)
        decision_task = create_decision_task(ticker)
           
        crew = Crew(
               agents=[quantitative_analyst, research_analyst, portfolio_manager],
               tasks=[quant_task, research_task, decision_task],
               process=Process.sequential,
               verbose=False,  # Disable verbose for API
               cache=False
           )
           
        result = crew.kickoff()
           
           # Parse the result
        result_str = str(result)
        result_str = result_str.replace('</r>', '').replace('<r>', '').strip()
        decision = "UNKNOWN"
        confidence = "UNKNOWN"
        risk = "UNKNOWN"
           
        for line in result_str.split("\n"):
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
           
           # Auto-downgrade confidence if missing data
        if "⚠️" in result_str or "MISSING DATA" in result_str.upper():
            if confidence == "HIGH":
                confidence = "MEDIUM"
            elif confidence == "MEDIUM":
                confidence = "LOW"
           
        return jsonify({
            'ticker': ticker,
            'decision': decision,
            'confidence': confidence,
            'risk_level': risk,
            'full_analysis': result_str
           })
       
    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})
   
if __name__ == '__main__':
       port = int(os.environ.get('PORT', 8080))
       app.run(host='0.0.0.0', port=port)