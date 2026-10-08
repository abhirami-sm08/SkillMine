"""
Urgency, Phishing & Fraud Keyword Dictionaries for ScamShield NLP Engine.
"""

# Categorized suspicious indicators with assigned risk weights (1 to 5 scale)

FINANCIAL_PATTERNS = {
    "registration fee": 5,
    "security deposit": 5,
    "processing fee": 5,
    "training fee": 4,
    "laptop fee": 4,
    "certificate fee": 4,
    "pay registration": 5,
    "upi payment": 4,
    "gpay": 3,
    "phonepe": 3,
    "crypto": 4,
    "usdt": 4,
    "refundable deposit": 4,
    "pay first": 5,
    "initial investment": 4,
    "buy equipment": 4,
}

COMMUNICATION_PATTERNS = {
    "whatsapp only": 4,
    "whatsapp group": 4,
    "telegram group": 5,
    "telegram channel": 5,
    "dm on instagram": 4,
    "dm me on whatsapp": 4,
    "contact on telegram": 5,
    "direct message telegram": 5,
    "personal email": 3,
}

URGENCY_PATTERNS = {
    "urgent hiring": 3,
    "immediate joiner": 3,
    "limited seats": 3,
    "apply within 2 hours": 4,
    "spot offer": 4,
    "no interview": 5,
    "direct selection": 5,
    "guaranteed selection": 5,
    "no resume required": 4,
    "no experience needed": 2,
    "earn per day": 4,
    "part time 1 hour": 4,
    "daily payout": 4,
    "easy online job": 4,
    "type captcha earn": 5,
    "data entry high pay": 4,
}

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "yandex.com",
    "protonmail.com", "icloud.com", "rediffmail.com", "zoho.com"
}


def analyze_urgency_and_suspicion(text: str, domain: str = "", email: str = "") -> dict:
    """
    Computes financial risk score, communication risk score, urgency risk score,
    and returns detected keyword matches and flags.
    """
    if not text:
        text = ""
    text_lower = text.lower()

    financial_matches = [kw for kw in FINANCIAL_PATTERNS if kw in text_lower]
    comm_matches = [kw for kw in COMMUNICATION_PATTERNS if kw in text_lower]
    urgency_matches = [kw for kw in URGENCY_PATTERNS if kw in text_lower]

    financial_score = sum(FINANCIAL_PATTERNS[kw] for kw in financial_matches)
    comm_score = sum(COMMUNICATION_PATTERNS[kw] for kw in comm_matches)
    urgency_score = sum(URGENCY_PATTERNS[kw] for kw in urgency_matches)

    # Email domain check
    is_free_email = False
    if email:
        email_domain = email.split("@")[-1].lower() if "@" in email else ""
        if email_domain in FREE_EMAIL_DOMAINS:
            is_free_email = True
            comm_score += 3
            comm_matches.append(f"Free email provider (@{email_domain})")

    # Unverified domain check
    is_suspicious_domain = False
    if domain:
        clean_domain = domain.lower().replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0]
        if any(free in clean_domain for free in ["wordpress.com", "wixsite.com", "blogspot.com", "weebly.com"]):
            is_suspicious_domain = True
            comm_score += 4
            comm_matches.append(f"Free hosted domain ({clean_domain})")

    total_risk_points = financial_score + comm_score + urgency_score
    # Normalize score 0.0 to 1.0 (cap at 25 points)
    normalized_urgency_index = min(1.0, total_risk_points / 25.0)

    return {
        "financial_matches": financial_matches,
        "communication_matches": comm_matches,
        "urgency_matches": urgency_matches,
        "financial_score": financial_score,
        "communication_score": comm_score,
        "urgency_score": urgency_score,
        "total_risk_points": total_risk_points,
        "urgency_index": round(normalized_urgency_index, 4),
        "is_free_email": is_free_email,
        "is_suspicious_domain": is_suspicious_domain
    }
