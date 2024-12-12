import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from model.api_caller import APICaller
from prompts.prompt_system import PromptSystem

index = faiss.read_index('faiss_db/review_index.faiss')
df = pd.read_csv('parsed_geo_reviews.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')
prompt_system = PromptSystem()
api_caller = APICaller(api_key='')

def generate_review_from_faiss(query, category, rating, keywords, num_similar_reviews=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), num_similar_reviews)
    similar_reviews = df.iloc[indices[0]]['text'].tolist()
    
    system_prompt = prompt_system.get_system_prompt()
    user_prompt = prompt_system.generate_user_prompt(category, rating, keywords, similar_reviews)
    
    return api_caller.call_gpt35_turbo(system_prompt, user_prompt)