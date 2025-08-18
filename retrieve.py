import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CompactBERTT5Inference:
    def __init__(self, model_dir, kb_csv):
        # Load trained model and tokenizer
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.t5_model = self.t5_model.to(device)
        
        # Load retrieval system and knowledge base
        self.retriever = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        self.knowledge_base = pd.read_csv(kb_csv)
        self.kb_embeddings = self.retriever.encode(self.knowledge_base['input'].tolist(), batch_size=32)
        
    def get_context(self, query, top_k=1):
        query_emb = self.retriever.encode([query])
        similarities = cosine_similarity(query_emb, self.kb_embeddings)[0]
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        return [
            {"question": self.knowledge_base.iloc[idx]['input'],
             "answer": self.knowledge_base.iloc[idx]['output'],
             "sim": similarities[idx]}
            for idx in top_idx if similarities[idx] > 0.3
        ]
        
    def answer_question(self, question):
        # Retrieve top 3 similar answers for richer context
        context = self.get_context(question, top_k=3)
        kb_context = "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in context]
        ) if context else ""
                
        if kb_context:
            inp = (
                f"Medical question: {question}\n"
                f"Relevant knowledge base answers:\n{kb_context}\n"
                "Please provide a detailed, accurate, and well-defined medical answer based on the above information."
            )
        else:
            inp = (
                f"Medical question: {question}\n"
                "Please provide a detailed, accurate, and well-defined medical answer."
            )
                
        # Encode input with proper truncation
        inputs = self.t5_tokenizer(
            inp, 
            return_tensors="pt", 
            # max_length=5120, 
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
                
        self.t5_model.eval()
        with torch.no_grad():
            # Keep model on device instead of moving to CPU
            outputs = self.t5_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10240,       # Increased
                num_beams=4,
                do_sample=False,          # Changed to False
                pad_token_id=self.t5_tokenizer.pad_token_id,
                eos_token_id=self.t5_tokenizer.eos_token_id,
                early_stopping=False,
                length_penalty=1.0,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
)
                
            model_answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        return model_answer

    def answer_question_with_retry(self, question, max_retries=3):
        """
        Answer with retry mechanism for incomplete responses
        """
        for attempt in range(max_retries):
            answer = self.answer_question(question)
            
            # Check if answer seems complete (ends with punctuation or has reasonable length)
            if (answer.strip().endswith(('.', '!', '?')) or 
                len(answer.strip()) > 100):  # Adjust threshold as needed
                return answer
            
            print(f"Attempt {attempt + 1}: Answer seems incomplete, retrying...")
            
        return answer  # Return the last attempt even if incomplete

if __name__ == "__main__":
    # Initialize the QA System
    qa_system = CompactBERTT5Inference(model_dir="./bert_t5_model", kb_csv="train.csv")
        
    print("Medical QA System (type 'exit' to quit)\n")
    print("Features:")
    print("- Improved generation parameters for complete answers")
    print("- Retry mechanism for incomplete responses")
    print("- Better handling of model memory\n")
    
    while True:
        user_question = input("Enter your medical question: ")
        if user_question.strip().lower() == "exit":
            print("Goodbye!")
            break
            
        print("Generating answer...")
        answer = qa_system.answer_question_with_retry(user_question)
        print(f"Answer: {answer}\n")
        print("-" * 50)