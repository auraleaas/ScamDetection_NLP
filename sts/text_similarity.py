import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
from indoNLP.preprocessing import replace_slang
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import AutoTokenizer, AutoModel
import torch



# Load dataset
dataset = pd.read_csv('https://drive.google.com/uc?id=1c50z3TtNR4DRHCjz4vsPvV6pYnnsem4V')

fraud_data = dataset[dataset['label'] == 'scam']


# Initialize Sastrawi
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()



# Preprocessing function
def preprocess_indonesian_text(text, remove_punct=False, remove_stops=False, apply_stemming=False):

    if pd.isna(text) or text == '':
        return ''
    
    # Convert ke lowercase
    text = text.lower()
    
    # Normalize slang menggunakan IndoNLP
    text = replace_slang(text)
    
    # Normalize whitespace (remove extra spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()  
        
    # Remove stopwords 
    if remove_stops:
        text = stopword_remover.remove(text)
    
    # Apply stemming
    if apply_stemming:
        text = stemmer.stem(text)
    
    return text


# Test preprocessing 
test_texts = [
    "Mba, tlg isikan pulsa 50rb ke nomor ini dulu ya",
    "Selamat! Anda menang hadiah 10jt. Klik link ini",
    "Bos, ini nomor baruku. Tolong pinjemin dulu 2jt yah"
]

# print("Preprocessing Examples (IndoNLP + Sastrawi):")
# print("=" * 80)
# for test_text in test_texts:
#     preprocessed = preprocess_indonesian_text(test_text)
#     print(f"Original:     {test_text}")
#     print(f"Preprocessed: {preprocessed}")
#     print("-" * 80)



# Load IndoBERT model and tokenizer
model_name = 'indobenchmark/indobert-base-p1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# get embeddings function
def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    # Tokenize teks
    encoded_input = tokenizer(texts, padding=True, truncation=True, 
                             max_length=512, return_tensors='pt')
    
    # Dapatkan output model
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling 
    attention_mask = encoded_input['attention_mask']
    token_embeddings = model_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    

    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return embeddings.numpy()


# Preprocess scam data texts
fraud_texts = fraud_data['text'].fillna('').tolist()
fraud_texts_processed = [preprocess_indonesian_text(text) for text in fraud_texts]

# Encode scam data texts
fraud_embeddings = get_embeddings(fraud_texts_processed)


# Compute Semantic Similarity function
def compute_semantic_similarity(input_text, top_k=10):
    
    # Preprocess input text 
    input_text_processed = preprocess_indonesian_text(input_text)

    # Encode input text menggunakan IndoBERT
    input_embedding = get_embeddings([input_text_processed])
    
    # Compute cosine similarity
    similarities = cosine_similarity(input_embedding, fraud_embeddings)[0]
    
    # get top-k indices
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": fraud_data['text'].iloc[idx],
            "similarity": float(similarities[idx])
        })
    
    # Return top k results
    return results
