
import pandas as pd
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Simple evaluation without external dependencies
class CompactEvaluator:
    def __init__(self, model_path="./bert_t5_model"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.bert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load KB for context
        try:
            self.kb = pd.read_csv("train.csv")
            self.kb_emb = self.bert.encode(self.kb['input'].tolist())
        except: self.kb = None
    
    def get_context(self, query):
        if self.kb is None: return []
        query_emb = self.bert.encode([query])
        sims = cosine_similarity(query_emb, self.kb_emb)[0]
        idx = np.argmax(sims)
        return [{'q': self.kb.iloc[idx]['input'], 'a': self.kb.iloc[idx]['output'][:100]}] if sims[idx] > 0.3 else []
    
    def generate(self, question):
        context = self.get_context(question)
        if context:
            inp = f"Example: Q: {context[0]['q']} A: {context[0]['a']} Now: {question}"
        else:
            inp = f"Medical question: {question}"
        
        inputs = self.tokenizer(inp, return_tensors="pt", max_length=200, truncation=True)
        
        # Use CPU for stable generation
        self.model.cpu()
        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_cpu['input_ids'], max_length=80, temperature=0.7, 
                do_sample=True, pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def simple_bleu(self, ref, pred):
        # Simple word overlap score
        ref_words = set(ref.lower().split())
        pred_words = set(pred.lower().split())
        if not pred_words: return 0.0
        return len(ref_words & pred_words) / len(pred_words)
    
    def semantic_sim(self, ref, pred):
        if not pred.strip(): return 0.0
        ref_emb = self.bert.encode([ref])
        pred_emb = self.bert.encode([pred])
        return cosine_similarity(ref_emb, pred_emb)[0][0]
    
    def evaluate(self, test_file, max_samples=20):
        print(f"Evaluating on {test_file}...")
        
        try:
            df = pd.read_csv(test_file).head(max_samples)
        except:
            print(f"{test_file} not found!")
            return None
        
        bleu_scores, sem_scores = [], []
        predictions, references = [], []
        
        for _, row in df.iterrows():
            question, reference = row['input'], row['output']
            prediction = self.generate(question)
            
            bleu_scores.append(self.simple_bleu(reference, prediction))
            sem_scores.append(self.semantic_sim(reference, prediction))
            predictions.append(prediction)
            references.append(reference)
        
        # Results
        avg_bleu = np.mean(bleu_scores)
        avg_sem = np.mean(sem_scores)
        overall = (avg_bleu * 0.4 + avg_sem * 0.6)
        
        print(f"\nRESULTS:")
        print(f"  Word Overlap Score: {avg_bleu:.3f}")
        print(f"  Semantic Similarity: {avg_sem:.3f}")
        print(f"  Overall Score: {overall:.3f}")
        
        if overall >= 0.6: print("EXCELLENT performance!")
        elif overall >= 0.4: print("GOOD performance")
        else: print("NEEDS IMPROVEMENT")
        
        # Show examples
        print(f"\n TOP 2 EXAMPLES:")
        best_indices = np.argsort(sem_scores)[-2:][::-1]
        
        for i, idx in enumerate(best_indices, 1):
            print(f"\n--- Example {i} ---")
            print(f"Q: {df.iloc[idx]['input'][:80]}...")
            print(f"Expected: {references[idx][:100]}...")
            print(f"Generated: {predictions[idx][:100]}...")
            print(f"Semantic Score: {sem_scores[idx]:.3f}")
        
        return overall

def main():
    print("Compact Model Evaluation")
    evaluator = CompactEvaluator()
    
    # Test on available files
    for test_file in ["val.csv", "test.csv"]:
        score = evaluator.evaluate(test_file)
        if score: print(f"\n{test_file}: {score:.3f}")
        print("-" * 40)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
