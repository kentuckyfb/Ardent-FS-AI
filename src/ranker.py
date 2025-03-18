from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_by_similarity(keywords, items):
    # Encode keywords and items
    keyword_embedding = model.encode(" ".join(keywords), convert_to_tensor=True)
    item_embeddings = model.encode(items, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(keyword_embedding, item_embeddings)
    ranked_indices = np.argsort(similarities[0].numpy())[::-1]

    # Return ranked items
    return [items[i] for i in ranked_indices]