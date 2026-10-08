from typing import Dict, Any, List
from src.rules import detect_payment_demand


class TrustScoreEngine:
    """
    Combines ML fraud probability with deterministic risk evidence.

    The final score is a risk estimate, not a guarantee:
      - 75-100 : LOW RISK
      - 50-74  : MODERATE RISK
      - 0-49   : HIGH RISK / FRAUDULENT

    Unlike the original version, there is NO hard 85% ceiling. A clean offer
    can score above 85, while different evidence produces different scores.
    """

    def compute_trust_score(
        self,
        predicted_class: str,
        class_probabilities: Dict[str, float],
        urgency_analysis: Dict[str, Any],
        is_anomaly: bool,
        anomaly_score: float,
        blacklist_matches: List[Dict[str, str]],
        raw_text: str = "",
        secondary_verification_passed: bool = False,
    ) -> Dict[str, Any]:

        p_genuine = float(class_probabilities.get("Genuine", 0.0))
        p_suspicious = float(class_probabilities.get("Suspicious", 0.0))
        p_fraud = float(class_probabilities.get("Fraudulent", 0.0))

        # Clamp/renormalize in case a caller sends imperfect probabilities.
        probs = {
            "Genuine": max(0.0, p_genuine),
            "Suspicious": max(0.0, p_suspicious),
            "Fraudulent": max(0.0, p_fraud),
        }
        total = sum(probs.values())
        if total <= 0:
            probs = {"Genuine": 1 / 3, "Suspicious": 1 / 3, "Fraudulent": 1 / 3}
        else:
            probs = {k: v / total for k, v in probs.items()}

        p_genuine = probs["Genuine"]
        p_suspicious = probs["Suspicious"]
        p_fraud = probs["Fraudulent"]

        explanation_alerts: List[str] = []
        penalties: List[str] = []
        bonuses: List[str] = []

        # ML model risk: fraud probability carries the most weight.
        # Suspicious probability contributes partial risk.
        model_risk = min(100.0, 100.0 * p_fraud + 55.0 * p_suspicious)

        # Rule-based evidence risk.
        fin_score = float(urgency_analysis.get("financial_score", 0) or 0)
        comm_score = float(urgency_analysis.get("communication_score", 0) or 0)
        urg_score = float(urgency_analysis.get("urgency_score", 0) or 0)

        financial_risk = min(40.0, fin_score * 6.0)
        communication_risk = min(20.0, comm_score * 3.0)
        urgency_risk = min(15.0, urg_score * 2.5)
        blacklist_risk = min(70.0, len(blacklist_matches) * 45.0)
        anomaly_risk = min(20.0, max(0.0, float(anomaly_score)) * 20.0) if is_anomaly else 0.0

        rule_risk = min(
            100.0,
            financial_risk
            + communication_risk
            + urgency_risk
            + blacklist_risk
            + anomaly_risk,
        )

        # Primary score: mostly the ML estimate, supported by deterministic evidence.
        combined_risk = 0.65 * model_risk + 0.35 * rule_risk
        trust_score = 100.0 - combined_risk

        # Strong deterministic controls override a model miss.
        payment_rule = detect_payment_demand(raw_text or " ".join(
            urgency_analysis.get("financial_matches", [])
        ))
        if payment_rule.triggered:
            trust_score = min(trust_score, payment_rule.safety_score_cap)
            penalties.append(
                "Deterministic payment-demand rule triggered: "
                + ", ".join(payment_rule.matches)
            )
            explanation_alerts.append(
                "CRITICAL: Explicit upfront payment demand detected; "
                "verification score capped at 30%."
            )

        if blacklist_matches:
            for match in blacklist_matches:
                penalties.append(
                    f"Blacklist Match: {match['type']} '{match['value']}' ({match['reason']})"
                )
                explanation_alerts.append(
                    f"CRITICAL: Blacklist match detected "
                    f"({match['type']}: {match['value']})."
                )

        if fin_score > 0:
            matches = ", ".join(urgency_analysis.get("financial_matches", []))
            penalties.append(f"Financial payment trigger detected: {matches}")
            explanation_alerts.append(
                f"Payment/fee language detected: {matches}. "
                "Legitimate employers normally do not require upfront job fees."
            )

        if comm_score > 0:
            matches = ", ".join(urgency_analysis.get("communication_matches", []))
            penalties.append(f"Unverified communication channel: {matches}")
            explanation_alerts.append(
                f"Recruitment communication risk detected: {matches}."
            )

        if urg_score > 0:
            matches = ", ".join(urgency_analysis.get("urgency_matches", []))
            penalties.append(f"High urgency / bypassed evaluation: {matches}")
            explanation_alerts.append(
                f"Pressure or weak screening signals detected: {matches}."
            )

        if is_anomaly:
            penalties.append(
                f"Statistical stipend/experience anomaly "
                f"(score: {float(anomaly_score):.2f})"
            )
            explanation_alerts.append(
                "Offer appears statistically unusual compared with the training distribution."
            )

        if financial_risk == 0:
            bonuses.append("No known financial-fee language detected.")
        if not urgency_analysis.get("is_free_email", False):
            bonuses.append("Recruiter email is not from a common free-email provider.")

        # Model confidence should be exposed, not artificially capped.
        max_confidence = max(probs.values())

        final_trust_score = round(max(0.0, min(100.0, trust_score)), 1)

        if final_trust_score >= 75.0:
            risk_level = "LOW RISK"
            status_color = "GREEN"
            recommendation = (
                "Low-risk result. Still verify the employer through its official "
                "website before sharing sensitive information."
            )
        elif final_trust_score >= 50.0:
            risk_level = "MODERATE RISK"
            status_color = "YELLOW"
            recommendation = (
                "Proceed with caution. Independently verify the company, recruiter "
                "identity, interview process, and never pay upfront fees."
            )
        else:
            risk_level = "HIGH RISK / FRAUDULENT"
            status_color = "RED"
            recommendation = (
                "Do not send money or sensitive credentials. Verify the employer "
                "through an official channel before taking any action."
            )

        # Evidence state prevents overclaiming when the model is uncertain.
        sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        margin = sorted_probs[0][1] - sorted_probs[1][1]
        if max_confidence < 0.55 or margin < 0.10:
            outcome_status = "Needs Manual Verification"
            explanation_alerts.append(
                "Model confidence is not sufficiently separated between classes; "
                "treat this result as uncertain."
            )
        else:
            outcome_status = "Supported by ML + Rule Signals"

        return {
            "trust_score": final_trust_score,
            "risk_level": risk_level,
            "status_color": status_color,
            "recommendation": recommendation,
            "penalties": penalties,
            "bonuses": bonuses,
            "explanation_alerts": explanation_alerts,
            "outcome_status": outcome_status,
            "max_confidence": round(float(max_confidence), 4),
            "class_probabilities": probs,
            "breakdown": {
                "model_risk": round(model_risk, 2),
                "rule_risk": round(rule_risk, 2),
                "financial_risk": round(financial_risk, 2),
                "communication_risk": round(communication_risk, 2),
                "urgency_risk": round(urgency_risk, 2),
                "blacklist_risk": round(blacklist_risk, 2),
                "anomaly_risk": round(anomaly_risk, 2),
                "payment_rule_cap": float(payment_rule.safety_score_cap)
                if payment_rule.triggered
                else 100.0,
            },
        }
