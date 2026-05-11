# retrieval.py
# TF-IDF + Cosine Similarity se FAQ match karta hai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import FAQ_DATA

# Threshold kam kiya — "forgot password" jaisi queries bhi match hongi
CONFIDENCE_THRESHOLD = 0.15


def preprocess(text: str) -> str:
    """Basic text cleaning"""
    return text.lower().strip()


def find_best_match(user_query: str) -> dict:
    """
    User query ke liye sabse relevant FAQ dhundta hai.

    Steps:
    1. FAQ question + answer dono combine karke TF-IDF vectorize karo
       (sirf question match karna kaafi nahi — synonyms miss ho jaate hain)
    2. Cosine similarity calculate karo
    3. Best match return karo ya fallback do
    """

    if not user_query or not user_query.strip():
        return {
            "answer": "Please provide a valid question.",
            "confidence": 0.0,
            "source": [],
            "fallback": True,
            "_retrieved_faq": None
        }

    # Question + Answer dono combine karo — better matching ke liye
    # Example: "forgot password" → "reset password follow email instructions" se match hoga
    faq_texts = [
        preprocess(item["question"] + " " + item["answer"])
        for item in FAQ_DATA
    ]
    cleaned_query = preprocess(user_query)

    # TF-IDF matrix banao — FAQ texts + user query ek saath
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    all_texts = faq_texts + [cleaned_query]

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return {
            "answer": "I don't have enough information to answer this.",
            "confidence": 0.0,
            "source": [],
            "fallback": True,
            "_retrieved_faq": None
        }

    # User query vector (last row) vs FAQ vectors (baaki sab)
    query_vector = tfidf_matrix[-1]
    faq_vectors = tfidf_matrix[:-1]

    # Cosine similarity calculate karo
    similarities = cosine_similarity(query_vector, faq_vectors).flatten()

    # Best match index aur score
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    # Agar score threshold se kam hai → fallback
    if best_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer this.",
            "confidence": round(best_score, 4),
            "source": [],
            "fallback": True,
            "_retrieved_faq": None
        }

    best_faq = FAQ_DATA[best_idx]

    return {
        "answer": best_faq["answer"],
        "confidence": round(best_score, 4),
        "source": [best_faq["id"]],
        "fallback": False,
        "_retrieved_faq": best_faq  # LLM step ke liye internal use (response mein nahi aayega)
    }