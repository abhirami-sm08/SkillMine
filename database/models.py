from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ScannedOffer(Base):
    """Stores scanned job/internship offers, predictions, and trust scores."""
    __tablename__ = 'scanned_offers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=False)
    company_domain = Column(String(255), nullable=True)
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    department = Column(String(100), nullable=False, default="General") # CS/IT, ECE, ME, CE, Biotech, Business
    location = Column(String(100), nullable=True, default="Remote")
    stipend = Column(Float, nullable=False, default=0.0)
    exp_years = Column(Float, nullable=False, default=0.0)
    description = Column(Text, nullable=False)
    posting_date = Column(DateTime, default=datetime.now)

    # Model Output Attributes
    predicted_class = Column(String(50), nullable=False) # Genuine, Suspicious, Fraudulent
    trust_score = Column(Float, nullable=False)
    archetype_cluster = Column(Integer, nullable=False, default=0)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_score = Column(Float, nullable=False, default=0.0)
    urgency_index = Column(Float, nullable=False, default=0.0)
    risk_factors_json = Column(Text, nullable=True) # JSON array of explanation alerts
    scanned_at = Column(DateTime, default=datetime.now)


class BlacklistEntity(Base):
    """Stores blacklisted domain names, company names, emails, phones, and payment handles."""
    __tablename__ = 'blacklist_entities'

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(50), nullable=False) # DOMAIN, COMPANY, EMAIL, PHONE, PAYMENT_HANDLE
    entity_value = Column(String(255), nullable=False, unique=True)
    reason = Column(String(255), nullable=False)
    added_at = Column(DateTime, default=datetime.now)


class AnalyticsLog(Base):
    """System activity & audit log for OLAP tracking."""
    __tablename__ = 'analytics_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(100), nullable=False) # SCAN, BLACKLIST_ADD, MODEL_RETRAIN
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
