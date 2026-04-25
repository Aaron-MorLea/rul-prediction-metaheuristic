import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class EngineAlert:
    """Engine alert data for LLM explanation."""
    engine_id: int
    predicted_rul: float
    risk_level: str
    maintenance_action: str
    current_cycle: float
    sensor_readings: Optional[Dict[str, float]] = None


class LLMAssistant:
    """
    LLM-powered assistant for RUL explanations and report generation.
    
    Uses LangChain for integration with OpenAI or local LLMs.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = None
        
        if self.api_key:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LangChain LLM."""
        try:
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key
            )
        except ImportError:
            print("LangChain not installed. Run: pip install langchain langchain-openai")
    
    def explain_alert(self, alert: EngineAlert) -> str:
        """
        Generate natural language explanation for an engine alert.
        """
        if self.llm is None:
            return self._generate_fallback_explanation(alert)
        
        prompt = f"""
        You are a reliability engineer assistant. Explain the following engine alert 
        in clear, professional language for maintenance technicians.
        
        Engine ID: {alert.engine_id}
        Predicted RUL: {alert.predicted_rul:.0f} cycles
        Risk Level: {alert.risk_level}
        Recommended Action: {alert.maintenance_action}
        Current Cycle: {alert.current_cycle:.0f}
        
        Provide a brief explanation (2-3 sentences) of:
        1. What this alert means
        2. Why this risk level was assigned
        3. What the technician should do next
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return self._generate_fallback_explanation(alert)
    
    def generate_daily_report(self, alerts: List[EngineAlert]) -> str:
        """
        Generate a daily maintenance report from multiple engine alerts.
        """
        if self.llm is None:
            return self._generate_fallback_report(alerts)
        
        alerts_summary = "\n".join([
            f"- Engine {a.engine_id}: RUL={a.predicted_rul:.0f} cycles, "
            f"Risk={a.risk_level}, Action={a.maintenance_action}"
            for a in alerts
        ])
        
        prompt = f"""
        You are a reliability engineering manager. Generate a concise daily 
        maintenance report from the following engine alerts.
        
        ALERTS:
        {alerts_summary}
        
        Format the report with:
        1. Summary paragraph (overall fleet health)
        2. Critical items requiring immediate attention
        3. Recommended maintenance schedule for the next 24-48 hours
        4. Resource allocation recommendations
        
        Keep it professional and actionable.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return self._generate_fallback_report(alerts)
    
    def answer_query(self, query: str, context: Dict) -> str:
        """
        Answer specific queries about engine fleet data.
        """
        if self.llm is None:
            return "LLM not configured. Please set OPENAI_API_KEY."
        
        context_str = json.dumps(context, indent=2)
        
        prompt = f"""
        You are a reliability engineering expert. Answer the following query 
        based on the provided fleet data.
        
        FLEET DATA:
        {context_str}
        
        QUERY: {query}
        
        Provide a clear, concise answer based on the data.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_fallback_explanation(self, alert: EngineAlert) -> str:
        """Generate basic explanation without LLM."""
        rul = alert.predicted_rul
        risk = alert.risk_level
        
        if risk == "CRITICAL":
            return (
                f"Engine {alert.engine_id} is in CRITICAL condition. "
                f"With only {rul:.0f} cycles of remaining useful life, "
                f"immediate maintenance is required. Stop operation and "
                f"schedule emergency inspection."
            )
        elif risk == "HIGH":
            return (
                f"Engine {alert.engine_id} shows HIGH risk with {rul:.0f} cycles "
                f"of RUL remaining. Plan maintenance within 24-48 hours. "
                f"Monitor closely for any degradation."
            )
        elif risk == "MEDIUM":
            return (
                f"Engine {alert.engine_id} has {rul:.0f} cycles of RUL remaining "
                f"and is at MEDIUM risk. Schedule maintenance during next "
                f"planned downtime."
            )
        else:
            return (
                f"Engine {alert.engine_id} is operating normally with {rul:.0f} "
                f"cycles of RUL remaining. Continue regular monitoring."
            )
    
    def _generate_fallback_report(self, alerts: List[EngineAlert]) -> str:
        """Generate basic report without LLM."""
        critical = [a for a in alerts if a.risk_level == "CRITICAL"]
        high = [a for a in alerts if a.risk_level == "HIGH"]
        medium = [a for a in alerts if a.risk_level == "MEDIUM"]
        low = [a for a in alerts if a.risk_level == "LOW"]
        
        report = "DAILY MAINTENANCE REPORT\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Fleet Overview: {len(alerts)} engines monitored\n"
        report += f"- Critical: {len(critical)}\n"
        report += f"- High: {len(high)}\n"
        report += f"- Medium: {len(medium)}\n"
        report += f"- Low: {len(low)}\n\n"
        
        if critical:
            report += "CRITICAL ACTIONS REQUIRED:\n"
            for a in critical:
                report += f"- Engine {a.engine_id}: RUL {a.predicted_rul:.0f} cycles\n"
            report += "\n"
        
        if high:
            report += "SCHEDULE WITHIN 24-48 HOURS:\n"
            for a in high:
                report += f"- Engine {a.engine_id}: RUL {a.predicted_rul:.0f} cycles\n"
            report += "\n"
        
        report += "RECOMMENDATION: Focus resources on critical and high-risk engines. "
        "Medium-risk engines can be scheduled for next planned maintenance window."
        
        return report


def create_llm_assistant(
    model_name: str = "gpt-4",
    api_key: Optional[str] = None
) -> LLMAssistant:
    """Factory function to create LLM assistant."""
    return LLMAssistant(model_name=model_name, api_key=api_key)


if __name__ == "__main__":
    assistant = create_llm_assistant()
    
    test_alert = EngineAlert(
        engine_id=1,
        predicted_rul=45.0,
        risk_level="HIGH",
        maintenance_action="SCHEDULE_SOON",
        current_cycle=180.0
    )
    
    print("Alert Explanation:")
    print(assistant.explain_alert(test_alert))
    print("\n" + "=" * 40 + "\n")
    
    alerts = [
        test_alert,
        EngineAlert(2, 120.0, "MEDIUM", "PLAN_NEXT", 200.0),
        EngineAlert(3, 15.0, "CRITICAL", "IMMEDIATE", 280.0)
    ]
    
    print("Daily Report:")
    print(assistant.generate_daily_report(alerts))