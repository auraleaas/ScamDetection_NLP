class SimilarityMatcher:
    def __init__(self, fraud_data, model, tokenizer):
        self.fraud_data = fraud_data
        self.model = model
        self.tokenizer = tokenizer
        self.fraud_texts = fraud_data['text'].fillna('').tolist()
        self.fraud_embeddings = self._get_embeddings_batch(self.fraud_texts)

    def _get_embeddings_batch(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._get_embeddings(batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def _get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True,
                                       max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        attention_mask = encoded_input['attention_mask']
        # Access the last hidden state from the hidden_states tuple
        token_embeddings = model_output.hidden_states[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                    torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings.cpu().numpy()

    def find_similar(self, text, top_k=3):
        from sklearn.metrics.pairwise import cosine_similarity
        input_embedding = self._get_embeddings([text])
        similarities = cosine_similarity(input_embedding, self.fraud_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                'text': self.fraud_texts[idx],
                'similarity': float(similarities[idx])
            })
        return results

similarity_matcher = SimilarityMatcher(fraud_data, model, tokenizer)
print("Similarity matcher ready!")