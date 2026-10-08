import os
import json
import pandas as pd
from typing import List, Dict, Any

class AssociationRuleMiner:
    """
    Association Rule Mining Engine for ScamShield (Apriori / FP-Growth via mlxtend).
    Mines strong behavioral rules such as {No Interview, Registration Fee} -> {Fraudulent}.
    """

    def __init__(self, min_support: float = 0.05, min_confidence: float = 0.5, min_lift: float = 1.2):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.mined_rules_df = pd.DataFrame()

    def prepare_transactions(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Discretizes job posting attributes into boolean itemsets.
        """
        transactions = []
        for _, row in df.iterrows():
            itemset = {}
            desc = str(row.get('description', '')).lower()
            email = str(row.get('contact_email', '')).lower()
            stipend = float(row.get('stipend', 0.0))
            exp = float(row.get('exp_years', 0.0))
            label_source = target_column if target_column in df.columns else None
            label = str(row.get(label_source) if label_source else (row.get('label') or row.get('predicted_class') or 'Genuine'))

            itemset['Fee_Required'] = any(kw in desc for kw in ['registration fee', 'security deposit', 'pay registration', 'processing fee', 'pay first'])
            itemset['No_Interview'] = any(kw in desc for kw in ['no interview', 'direct selection', 'guaranteed selection', 'spot offer'])
            itemset['Telegram_WhatsApp_Contact'] = any(kw in desc for kw in ['telegram group', 'whatsapp only', 'telegram channel', 'dm on instagram'])
            itemset['Free_Email_Provider'] = any(kw in email for kw in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])
            itemset['Unrealistic_Salary'] = (stipend > 80000 and exp <= 1.0)
            itemset['Urgent_Hiring_Claim'] = any(kw in desc for kw in ['urgent hiring', 'immediate joiner', 'limited seats', 'apply within 2 hours'])
            
            # Label item
            itemset['Class_Fraudulent'] = (label == 'Fraudulent')
            itemset['Class_Suspicious'] = (label == 'Suspicious')
            itemset['Class_Genuine'] = (label == 'Genuine')

            transactions.append(itemset)

        return pd.DataFrame(transactions)

    def mine_rules(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Runs Apriori or FP-Growth to discover frequent itemsets and mine association rules.
        """
        trans_df = self.prepare_transactions(df, target_column=target_column)

        try:
            from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
            
            # Use FP-Growth if available for speed, else Apriori
            try:
                frequent_itemsets = fpgrowth(trans_df, min_support=self.min_support, use_colnames=True)
            except Exception:
                frequent_itemsets = apriori(trans_df, min_support=self.min_support, use_colnames=True)

            if frequent_itemsets.empty:
                self.mined_rules_df = pd.DataFrame()
                return self.mined_rules_df

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
            
            # Filter rules by min_lift
            rules = rules[rules['lift'] >= self.min_lift]

            # Format antecedents and consequents into readable strings
            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: " + ".join(list(x)))
            rules['consequents_str'] = rules['consequents'].apply(lambda x: " + ".join(list(x)))
            
            rules = rules.sort_values(by="lift", ascending=False)
            self.mined_rules_df = rules
            return rules

        except ImportError as exc:
            raise RuntimeError("Association rule mining requires mlxtend; install it instead of using mock fallback rules.") from exc
        except Exception:
            self.mined_rules_df = pd.DataFrame()
            return self.mined_rules_df

    def get_top_fraud_rules(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Returns top N rules leading to Fraudulent or Suspicious outcomes."""
        if self.mined_rules_df.empty:
            return []

        df = self.mined_rules_df.copy()
        fraud_rules = df[df['consequents_str'].str.contains('Class_Fraudulent|Class_Suspicious', case=False, na=False)]
        
        output = []
        for _, row in fraud_rules.head(top_n).iterrows():
            output.append({
                "antecedents": row['antecedents_str'],
                "consequents": row['consequents_str'],
                "support": round(float(row['support']), 4),
                "confidence": round(float(row['confidence']), 4),
                "lift": round(float(row['lift']), 4)
            })

        return output
