from sts.text_similarity import compute_semantic_similarity

query = "Halo Kak, kami dari admin grup 'Info Crypto Valid'. Anda harus bayar biaya langganan bulanan 200rb untuk tetap dapat info coin yang bakal pump."

results = compute_semantic_similarity(query, top_k=10)
print(results)
