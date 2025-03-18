from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load models
kw_model = KeyBERT()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_keywords(query):
    """
    Extracts keywords using KeyBERT.
    """
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5)
    return list(set([kw[0] for kw in keywords]))  # Remove duplicates

def prioritize_keywords(query, keywords):
    """
    Prioritizes keywords based on similarity to the query.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    keyword_embeddings = sentence_model.encode(keywords, convert_to_tensor=True)

    similarities = cosine_similarity(query_embedding.reshape(1, -1), keyword_embeddings)[0]
    return [kw for _, kw in sorted(zip(similarities, keywords), reverse=True)]

def generate_related_keywords(keywords):
    """
    Finds related keywords by checking cosine similarity between extracted keywords.
    """
    keyword_embeddings = sentence_model.encode(keywords)

    related_keywords = {}
    for i, keyword in enumerate(keywords):
        similarities = cosine_similarity([keyword_embeddings[i]], keyword_embeddings)[0]
        similar_words = [keywords[j] for j in np.argsort(similarities)[::-1] if j != i][:3]  # Get top 3 related keywords
        related_keywords[keyword] = similar_words
    
    return related_keywords

# Run the optimized functions
# if __name__ == "__main__":
#     query = "Give me all invoices from January to March 2025"
#     keywords = generate_keywords(query)
#     print("Generated Keywords:", keywords)

#     prioritized_keywords = prioritize_keywords(query, keywords)
#     print("Prioritized Keywords:", prioritized_keywords)