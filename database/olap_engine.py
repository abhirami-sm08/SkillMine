import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database.models import ScannedOffer, BlacklistEntity
from database.db_config import SessionLocal

class OLAPAnalyticsEngine:
    """
    OLAP Data Mining Analytics Engine for ScamShield.
    Implements Data Mining OLAP operations:
    1. Roll-up (Daily -> Weekly -> Monthly -> Departmental summary)
    2. Drill-down (Department -> Role / Fraud Subtype)
    3. Slice (Filter by single dimension, e.g. Dept='CS/IT')
    4. Dice (Multi-dimensional sub-cube filter, e.g. Dept + Fraud Type + Stipend Range)
    5. Pivot (Department vs. Fraud Type Cross-tabulation)
    """

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SessionLocal()

    def load_offers_dataframe(self) -> pd.DataFrame:
        """Loads all scanned offer records into Pandas DataFrame for OLAP computations."""
        offers = self.session.query(ScannedOffer).all()
        if not offers:
            return pd.DataFrame()

        data = []
        for o in offers:
            data.append({
                "id": o.id,
                "title": o.title,
                "company_name": o.company_name,
                "company_domain": o.company_domain,
                "contact_email": o.contact_email,
                "department": o.department,
                "location": o.location,
                "stipend": o.stipend,
                "exp_years": o.exp_years,
                "description": o.description,
                "posting_date": o.posting_date,
                "predicted_class": o.predicted_class,
                "trust_score": o.trust_score,
                "archetype_cluster": o.archetype_cluster,
                "is_anomaly": o.is_anomaly,
                "anomaly_score": o.anomaly_score,
                "urgency_index": o.urgency_index,
                "scanned_at": o.scanned_at
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['posting_date'] = pd.to_datetime(df['posting_date'])
            df['year_month'] = df['posting_date'].dt.to_period('M').astype(str)
            df['date'] = df['posting_date'].dt.date
        return df

    def rollup_time_series(self, freq: str = 'W') -> pd.DataFrame:
        """
        OLAP Roll-up operation: Aggregates scam counts by time frequency ('D' Daily, 'W' Weekly, 'M' Monthly).
        """
        df = self.load_offers_dataframe()
        if df.empty:
            return pd.DataFrame()

        df_grouped = df.groupby([pd.Grouper(key='posting_date', freq=freq), 'predicted_class']).size().unstack(fill_value=0)
        return df_grouped.reset_index()

    def drilldown_department(self, department_name: str) -> pd.DataFrame:
        """
        OLAP Drill-down operation: Drills into a specific department to analyze job title & company vulnerability.
        """
        df = self.load_offers_dataframe()
        if df.empty:
            return pd.DataFrame()

        dept_df = df[df['department'].str.upper() == department_name.upper()]
        if dept_df.empty:
            return pd.DataFrame()

        summary = dept_df.groupby(['title', 'predicted_class']).agg(
            total_offers=('id', 'count'),
            avg_trust_score=('trust_score', 'mean'),
            avg_stipend=('stipend', 'mean')
        ).reset_index()

        summary['avg_trust_score'] = summary['avg_trust_score'].round(1)
        summary['avg_stipend'] = summary['avg_stipend'].round(0)
        return summary

    def slice_data(self, dimension: str, value: Any) -> pd.DataFrame:
        """
        OLAP Slice operation: Filters entire data cube on a single dimension (e.g. predicted_class='Fraudulent').
        """
        df = self.load_offers_dataframe()
        if df.empty or dimension not in df.columns:
            return pd.DataFrame()

        return df[df[dimension] == value]

    def dice_subcube(self, departments: List[str] = None, fraud_types: List[str] = None,
                     min_trust: float = 0.0, max_trust: float = 100.0) -> pd.DataFrame:
        """
        OLAP Dice operation: Extracts sub-cube across multiple dimensions (Department, Fraud Type, Trust Score Range).
        """
        df = self.load_offers_dataframe()
        if df.empty:
            return pd.DataFrame()

        filtered = df[(df['trust_score'] >= min_trust) & (df['trust_score'] <= max_trust)]

        if departments:
            filtered = filtered[filtered['department'].isin(departments)]

        if fraud_types:
            filtered = filtered[filtered['predicted_class'].isin(fraud_types)]

        return filtered

    def pivot_dept_vs_fraud(self) -> pd.DataFrame:
        """
        OLAP Pivot operation: Generates Department vs. Fraud Classification contingency table.
        """
        df = self.load_offers_dataframe()
        if df.empty:
            return pd.DataFrame()

        pivot_table = pd.crosstab(df['department'], df['predicted_class'], margins=True, margins_name='Total')
        return pivot_table
