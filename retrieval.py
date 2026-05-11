# retrieval.py
# Matches FAQs using TF-IDF and Cosine Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import FAQ_DATA

# Lowered the threshold so queries like "forgot password" can also be matched
CONFIDENCE_THRESHOLD = 0.15


def preprocess(text: str) -> str:
    """Basic text cleaning"""
    return text.lower().strip()


def find_best_match(user_query: str) -> dict:
    """
    Finds the most relevant FAQ for the user's query.

    Steps:
    1. Combine both FAQ questions and answers, then apply TF-IDF vectorization
    (matching only questions is not enough — synonyms and related context may be missed)
    2. Calculate cosine similarity
    3. Return the best match or provide a fallback response
    """

    if not user_query or not user_query.strip():
        return {
            "answer": "Please provide a valid question.",
            "confidence": 0.0,
            "source": [],
            "fallback": True,
            "_retrieved_faq": None
        }

    # Combine both the question and answer for better matching
    # Example: "forgot password" can match with
    # "reset password by following the email instructions"
    faq_texts = [
        preprocess(item["question"] + " " + item["answer"])
        for item in FAQ_DATA
    ]
    cleaned_query = preprocess(user_query)

    # Create a TF-IDF matrix using both the FAQ texts and the user query together
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

    # Compare the user query vector (last row) with all FAQ vectors (remaining rows)
    query_vector = tfidf_matrix[-1]
    faq_vectors = tfidf_matrix[:-1]

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, faq_vectors).flatten()

    # Get the index and similarity score of the best matching result
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    # If the score is below the threshold, return a fallback response
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
        "_retrieved_faq": best_faq  # Used internally for the LLM step (will not be included in the response)
    }