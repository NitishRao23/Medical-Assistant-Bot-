#!/usr/bin/env python3
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(4)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CompactBERTT5:
    def __init__(self):
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        # Move model to device after loading
        self.t5_model = self.t5_model.to(device)
        self.kb_embeddings = None
        self.knowledge_base = None
    
    def load_kb(self, csv_path):
        self.knowledge_base = pd.read_csv(csv_path)
        questions = self.knowledge_base['input'].tolist()
        self.kb_embeddings = self.retriever.encode(questions, batch_size=32)
    
    def get_context(self, query, top_k=1):
        if self.kb_embeddings is None: return []
        query_emb = self.retriever.encode([query])
        similarities = cosine_similarity(query_emb, self.kb_embeddings)[0]
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        return [{'question': self.knowledge_base.iloc[idx]['input'], 
                'answer': self.knowledge_base.iloc[idx]['output'], 
                'sim': similarities[idx]} 
               for idx in top_idx if similarities[idx] > 0.3]
    
    def enhance_data(self, df, max_samples=50):
        enhanced = []
        for _, row in df.head(max_samples).iterrows():
            context = self.get_context(row['input'])
            if context and context[0]['sim'] > 0.4:
                inp = f"Example: Q: {context[0]['question']} A: {context[0]['answer']} Now: {row['input']}"
            else:
                inp = f"Medical question: {row['input']}"
            enhanced.append({'input': inp, 'output': row['output']})
        return enhanced

def tokenize(examples, tokenizer):
    model_inputs = tokenizer(examples["input"], max_length=200, padding=False, truncation=True)
    labels = tokenizer(examples["output"], max_length=50, padding=False, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    
    # Initialize system
    system = CompactBERTT5()
    system.load_kb("train.csv")
    
    # Load and enhance data
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    
    train_data = system.enhance_data(train_df, 50)
    val_data = system.enhance_data(val_df, 20)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data).map(
        lambda x: tokenize(x, system.t5_tokenizer), batched=True, batch_size=16
    ).remove_columns(["input", "output"])
    
    val_dataset = Dataset.from_list(val_data).map(
        lambda x: tokenize(x, system.t5_tokenizer), batched=True, batch_size=16
    ).remove_columns(["input", "output"])
    
    # Training setup
    training_args = TrainingArguments(
        output_dir="./bert_t5_results", learning_rate=5e-4,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        num_train_epochs=2, warmup_steps=10, save_strategy="epoch",
        eval_strategy="epoch", logging_steps=5, dataloader_num_workers=0,
        use_mps_device=True, load_best_model_at_end=True, report_to=None
    )
    
    trainer = Trainer(
        model=system.t5_model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=system.t5_tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=system.t5_tokenizer, model=system.t5_model, padding=True)
    )
    
    # Train and save
    trainer.train()
    trainer.save_model("./bert_t5_model")
    system.t5_tokenizer.save_pretrained("./bert_t5_model")
    
    # Test - Fixed MPS device issues
    system.t5_model.eval()  # Set to eval mode
    for question in ["What are the symptoms of diabetes?", "How to treat hypertension?"]:
        context = system.get_context(question)
        if context and context[0]['sim'] > 0.4:
            inp = f"Similar: {context[0]['question']} Answer: {context[0]['answer']} Now: {question}"
        else:
            inp = f"Medical question: {question}"
        
        inputs = system.t5_tokenizer(inp, return_tensors="pt", max_length=200, truncation=True)
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use CPU for generation to avoid MPS issues
            system.t5_model.cpu()
            inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
            outputs = system.t5_model.generate(
                inputs_cpu['input_ids'], 
                max_length=50, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=system.t5_tokenizer.pad_token_id
            )
            system.t5_model.to(device)  # Move back to device
            
        answer = system.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Q: {question}\nA: {answer}\n")
    
    print(" Training completed!")

if __name__ == "__main__":
    main()
