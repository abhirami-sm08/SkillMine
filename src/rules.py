import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


NEGATION_RE = re.compile(
    r"\b(no|not|never|without|waived|free|zero)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(registration|processing|training|security|laptop|certificate)?\s*"
    r"(fee|fees|deposit|charge|payment|required|needed)\b",
    re.IGNORECASE,
)

CURRENCY_AMOUNT_RE = re.compile(
    r"(?:\b(?:rs|inr|rupees|₹)\.?\s*\d[\d,]*(?:\.\d+)?\b|\b\d[\d,]*(?:\.\d+)?\s*(?:rs|inr|rupees|₹)\b)",
    re.IGNORECASE,
)

PAYMENT_KEYWORD_RE = re.compile(
    r"\b("
    r"pay|paid|payment|deposit|remit|transfer|send|gpay|googlepay|phonepe|upi|"
    r"registration\s+fee|processing\s+fee|training\s+fee|security\s+deposit|"
    r"laptop\s+(?:fee|deposit)|certificate\s+fee|documentation\s+charge"
    r")\b",
    re.IGNORECASE,
)

PAYMENT_HANDLE_RE = re.compile(r"\b[a-z0-9._-]{2,}@[a-z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?91[-\s]?)?[6-9]\d{9}\b")


@dataclass(frozen=True)
class PaymentRuleResult:
    triggered: bool
    fraud_score_floor: float
    safety_score_cap: float
    matches: List[str]
    reason: str


def detect_payment_demand(text: str) -> PaymentRuleResult:
    """Detect explicit upfront payment demands with negation guardrails."""
    text = text or ""
    lowered = text.lower()
    if NEGATION_RE.search(lowered):
        return PaymentRuleResult(False, 0.0, 100.0, [], "Payment language is negated.")

    matches = sorted(set(m.group(0).strip() for m in PAYMENT_KEYWORD_RE.finditer(text)))
    amounts = sorted(set(m.group(0).strip() for m in CURRENCY_AMOUNT_RE.finditer(text)))
    handles = sorted(set(m.group(0).strip() for m in PAYMENT_HANDLE_RE.finditer(text)))

    has_fee_phrase = any("fee" in m.lower() or "deposit" in m.lower() or "charge" in m.lower() for m in matches)
    has_transfer_action = any(m.lower() in {"pay", "paid", "payment", "deposit", "remit", "transfer", "send"} for m in matches)
    triggered = bool((amounts and (has_fee_phrase or has_transfer_action)) or (handles and has_fee_phrase))

    evidence = matches + amounts + handles
    return PaymentRuleResult(
        triggered=triggered,
        fraud_score_floor=70.0 if triggered else 0.0,
        safety_score_cap=30.0 if triggered else 100.0,
        matches=evidence,
        reason="Explicit upfront payment demand found." if triggered else "No explicit payment demand found.",
    )


def extract_phone_numbers(text: str) -> List[str]:
    return sorted(set(m.group(0).replace(" ", "").replace("-", "") for m in PHONE_RE.finditer(text or "")))


def find_blacklist_matches(
    *,
    text: str,
    domain: str = "",
    email: str = "",
    company_name: str = "",
    phone: str = "",
    blacklist_entries: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Match domains, emails, companies, phones, and payment handles against the active blacklist."""
    haystack = " ".join([text or "", domain or "", email or "", company_name or "", phone or ""]).lower()
    normalized_phones = set(extract_phone_numbers(haystack))
    matched = []

    for entry in blacklist_entries:
        value = str(entry.get("value") or entry.get("entity_value") or "").strip()
        entry_type = str(entry.get("type") or entry.get("entity_type") or "").upper()
        if not value:
            continue
        normalized_value = value.lower()
        phone_value = value.replace(" ", "").replace("-", "")
        if normalized_value in haystack or (entry_type == "PHONE" and phone_value in normalized_phones):
            matched.append(
                {
                    "type": entry_type or "UNKNOWN",
                    "value": value,
                    "reason": str(entry.get("reason", "")),
                }
            )

    return matched
