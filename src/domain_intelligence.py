from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, Optional


KNOWN_CORPORATE_DOMAINS = {
    "infosys": "infosys.com",
    "tcs": "tcs.com",
    "wipro": "wipro.com",
    "accenture": "accenture.com",
    "techmahindra": "techmahindra.com",
}

HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "@": "a"})


@dataclass(frozen=True)
class DomainIntelligenceResult:
    domain: str
    is_new_domain: bool
    lookalike_target: Optional[str]
    recruiter_domain_matches_company: bool
    mca_status: str
    alerts: list


def normalize_domain(value: str) -> str:
    return (value or "").lower().replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0].strip()


def recruiter_domain_matches(email: str, company_domain: str) -> bool:
    email_domain = normalize_domain(email.split("@")[-1] if "@" in (email or "") else "")
    return bool(email_domain and normalize_domain(company_domain) and email_domain == normalize_domain(company_domain))


def detect_lookalike_domain(domain: str, known_domains: Optional[Dict[str, str]] = None) -> Optional[str]:
    known_domains = known_domains or KNOWN_CORPORATE_DOMAINS
    normalized = normalize_domain(domain).translate(HOMOGLYPHS)
    first_label = normalized.split(".")[0]
    for brand, official in known_domains.items():
        official_norm = normalize_domain(official).translate(HOMOGLYPHS)
        if brand in first_label and normalized != official_norm:
            return official
        score = SequenceMatcher(None, normalized, official_norm).ratio()
        if normalized != official_norm and score >= 0.72:
            return official
    return None


def check_mca_registry(company_name: str) -> str:
    """Stub for MCA API integration; returns an explicit unverified state."""
    return "UNVERIFIED_MCA_LOOKUP_NOT_CONFIGURED" if company_name else "NO_COMPANY_SUPPLIED"


def inspect_domain(
    domain: str,
    recruiter_email: str = "",
    company_name: str = "",
    company_domain: str = "",
    created_at: Optional[datetime] = None,
) -> DomainIntelligenceResult:
    normalized = normalize_domain(domain)
    alerts = []
    is_new = False
    if created_at is not None:
        age_days = (datetime.now(timezone.utc) - created_at.astimezone(timezone.utc)).days
        is_new = age_days < 90
        if is_new:
            alerts.append(f"Domain is newly registered ({age_days} days old).")

    lookalike = detect_lookalike_domain(normalized)
    if lookalike:
        alerts.append(f"Possible lookalike of {lookalike}.")

    matches = recruiter_domain_matches(recruiter_email, company_domain or domain)
    if recruiter_email and not matches:
        alerts.append("Recruiter email domain does not match the supplied company domain.")

    mca_status = check_mca_registry(company_name)
    if mca_status.startswith("UNVERIFIED"):
        alerts.append("MCA registry verification is not configured.")

    return DomainIntelligenceResult(normalized, is_new, lookalike, matches, mca_status, alerts)
