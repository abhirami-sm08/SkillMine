import re
import string
from typing import List, Dict, Any

# Try importing spaCy or NLTK with robust fallback
_SPACY_NLP = None
_NLTK_STOPWORDS = None

def _init_nlp():
    global _SPACY_NLP, _NLTK_STOPWORDS
    if _SPACY_NLP is not None and _NLTK_STOPWORDS is not None:
        return

    try:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            # Fallback if model not downloaded
            _SPACY_NLP = spacy.blank("en")
    except ImportError:
        _SPACY_NLP = None

    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _NLTK_STOPWORDS = set(stopwords.words("english"))
        except Exception:
            try:
                nltk.download("stopwords", quiet=True)
                _NLTK_STOPWORDS = set(stopwords.words("english"))
            except Exception:
                _NLTK_STOPWORDS = {
                    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
                    "was", "were", "with", "you", "your", "we", "our"
                }
    except Exception:
        _NLTK_STOPWORDS = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
            "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
            "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves"
        }

class TextCleaner:
    """
    NLP Text Preprocessing Pipeline for ScamShield.
    Handles text cleaning, link/email removal, stop-word filtering, and lemmatization.
    """

    def __init__(self):
        _init_nlp()
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+|bit\.ly/\S+|wa\.me/\S+|t\.me/\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract structural artifacts before stripping them."""
        if not text:
            return {"url_count": 0, "email_count": 0, "phone_count": 0, "urls": [], "emails": [], "phones": []}

        urls = self.url_pattern.findall(text)
        emails = self.email_pattern.findall(text)
        phones = self.phone_pattern.findall(text)

        return {
            "url_count": len(urls),
            "email_count": len(emails),
            "phone_count": len(phones),
            "urls": urls,
            "emails": emails,
            "phones": phones
        }

    def clean_text(self, text: str, keep_numbers: bool = False) -> str:
        """
        Full text cleaning pipeline:
        1. Lowercasing
        2. Removing URLs, Emails, Phone numbers
        3. Removing special characters & punctuation
        4. Lemmatization & Stop-word removal
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        cleaned = text.lower()

        # 2. Remove URLs, Emails, Phone numbers
        cleaned = self.url_pattern.sub(' ', cleaned)
        cleaned = self.email_pattern.sub(' ', cleaned)
        cleaned = self.phone_pattern.sub(' ', cleaned)

        # 3. Remove punctuation & special chars
        if keep_numbers:
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
        else:
            cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)

        # 4. Lemmatize & remove stop words
        tokens = cleaned.split()
        
        if _SPACY_NLP and hasattr(_SPACY_NLP, 'has_pipe') and _SPACY_NLP.has_pipe("lemmatizer"):
            try:
                doc = _SPACY_NLP(" ".join(tokens))
                lemmatized_tokens = [
                    token.lemma_.strip() for token in doc 
                    if not token.is_stop and len(token.lemma_.strip()) > 2
                ]
                return " ".join(lemmatized_tokens)
            except Exception:
                pass

        # Fallback to NLTK stop-words / simple token filtering
        stop_words = _NLTK_STOPWORDS or set()
        filtered_tokens = [
            w.strip() for w in tokens
            if w.strip() not in stop_words and len(w.strip()) > 2
        ]
        return " ".join(filtered_tokens)

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        return cleaned.split()
