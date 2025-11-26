from lime.lime_text import LimeTextExplainer
import numpy as np

class FraudClassifierForXAI:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def predict_proba(self, texts):
        '''Required format for LIME - with batch processing to save memory'''
        texts_cleaned = [clean_text(text) for text in texts]

        batch_size = 32  # reduced from default
        all_probs = []

        for i in range(0, len(texts_cleaned), batch_size):
            batch = texts_cleaned[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt',
                                   padding='max_length', truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            all_probs.append(probs.cpu().numpy())

            # clear cache after each batch
            del inputs, outputs, probs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return np.vstack(all_probs)

fraud_classifier_xai = FraudClassifierForXAI(model, tokenizer, device)

class FraudReasoningXAI:
    def __init__(self, classifier_pipeline, tokenizer, class_names=['not scam', 'neutral', 'scam']):
        self.classifier_pipeline = classifier_pipeline
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)

    def _predict_proba_wrapper(self, texts):
        return fraud_classifier_xai.predict_proba(texts)

    def explain_text(self, text_input, num_features=5, num_samples=5000):
        # generate explanation menggunakan LIME with reduced samples
        exp = self.explainer.explain_instance(
            text_input,
            self._predict_proba_wrapper,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1
        )

        top_label_idx = exp.top_labels[0]
        predicted_label = self.class_names[top_label_idx]

        explanation_list = exp.as_list(label=top_label_idx)
        keywords = [word for word, weight in explanation_list if weight > 0]

        return self._generate_natural_language_reasoning(predicted_label, keywords, text_input)

    def _generate_natural_language_reasoning(self, label, keywords, original_text):
        if not keywords:
            return f"Sistem mengklasifikasikan pesan ini sebagai **{label.upper()}**, namun pola spesifik sulit ditentukan secara individu."

        keywords_str = ", ".join([f"'{k}'" for k in keywords])

        reasoning = ""
        if label == 'scam':
            reasoning = (
                f"Pesan terdeteksi sebagai **SCAM**. \n"
                f"**Analisis:** Sistem menemukan indikasi penipuan kuat berdasarkan penggunaan kata kunci: **{keywords_str}**. \n"
                f"Kata-kata ini sering muncul dalam pola penipuan (seperti permintaan uang, urgensi, atau hadiah palsu) pada dataset historis."
            )
        elif label == 'neutral':
            reasoning = (
                f"Pesan dikategorikan **NETRAL**. \n"
                f"Meskipun mengandung kata **{keywords_str}**, konteksnya tidak cukup kuat untuk dianggap sebagai penipuan atau pesan resmi yang valid."
            )
        else: # Not Scam / Info Resmi
            reasoning = (
                f"Pesan ini diklasifikasikan **AMAN (Not Scam)**. \n"
                f"Struktur kalimat dan kata kunci seperti **{keywords_str}** konsisten dengan format informasi atau komunikasi wajar."
            )

        return reasoning

# Inisialisasi Explainer
xai_module = FraudReasoningXAI(
    classifier_pipeline=fraud_classifier_xai.predict_proba,
    tokenizer=tokenizer
)

# Teks Query (Contoh kasus)
query_text = "Halo Kak, kami dari admin grup 'Info Crypto Valid'. Anda harus bayar biaya langganan bulanan 200rb untuk tetap dapat info coin yang bakal pump."

print("Generating LIME explanation (this may take 30-60 seconds on CPU)...")

# Jalankan Reasoning with optimized parameters
reasoning_output = xai_module.explain_text(query_text, num_features=5, num_samples=10)

print("="*50)
print("HASIL TEST xAI REASONING")
print("="*50)
print(f"Input Text: {query_text}\n")
print(reasoning_output)