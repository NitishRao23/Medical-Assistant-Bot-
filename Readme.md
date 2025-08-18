**MEDICAL ASSISTANT BOT PROJECT**

**🏥 Project Overview**
Medical chatbot is an intelligent question-answering system that can understand and respond to user queries about medical diseases and health topics. The system uses a hybrid approach combining BERT-based retrieval for finding relevant medical information and T5-based generation for creating comprehensive answers. The project includes complete data preprocessing, model training using NLP techniques, performance evaluation, and a full-stack web application with FastAPI backend and Streamlit frontend. Users can ask medical questions through a user-friendly interface and receive detailed, contextually relevant answers based on the trained knowledge base.

**Porblems and Solutions **
**
Porblems:**
•	Users often struggle to find quick, reliable answers to basic medical questions, leading to unnecessary anxiety or delayed healthcare decisions
•	Patients spend hours searching through multiple websites and medical resources to understand their symptoms or conditions
•	Healthcare professionals face information overload when trying to quickly reference medical knowledge during consultations
•	Internet searches often return unreliable or conflicting medical information from non-authoritative sources


*Solutions: *
•	Provides immediate, accurate responses to medical queries using trained NLP models on curated medical datasets
•	 Offers 24/7 availability for medical information access without waiting for appointments or professional consultations
•	 Uses hybrid BERT+T5 architecture to understand complex medical questions and generate contextually relevant answers
•	 Implements semantic search to find most relevant medical information from comprehensive knowledge base
•	 Delivers consistent, reliable medical information based on structured datasets and validated medical knowledge

** Current Limitations & Technical Specifications**

** Limitations in Current Medical Information Systems**

The current landscape of medical question-answering systems and healthcare chatbots primarily focuses on appointment scheduling, basic symptom checking, or managing existing patient records. These systems often lack the capability to provide comprehensive, contextually-aware answers to diverse medical queries. Most existing solutions either:

•	Rely on simple keyword matching without understanding query context
•	Provide generic responses that lack medical accuracy and depth
•	Require users to navigate complex medical databases manually
•	Cannot handle varied question formats or provide detailed explanations

In contrast, our medical question-answering system addresses this critical gap by focusing on intelligent medical information retrieval and generation. We have leveraged advanced NLP techniques, combining BERT-based semantic retrieval with T5-based answer generation, to develop a system capable of comprehensively understanding and responding to complex medical queries with accurate, contextually relevant information.

 Technical Approach: 

•	Advanced NLP Implementation: Utilize state-of-the-art language models (BERT + T5) for accurate comprehension of diverse medical queries and generation of detailed, medically sound responses
•	Intelligent Knowledge Base Integration: Implement semantic search capabilities with continuously accessible medical knowledge base to ensure relevant information retrieval for query processing
•	Hybrid Model Architecture: Develop and evaluate retrieval-augmented generation approach to identify the most effective combination of semantic search and answer generation models
•	Scalable Web Application Integration: Ensure seamless deployment through FastAPI backend and Streamlit frontend for real-time medical question processing and user interaction
•	User-Centric Interface Design: Provide intuitive, accessible web interface that enables users to easily submit medical queries and receive comprehensive answers, enhancing overall user experience and medical information accessibility.

Architecture Framework: 

![Medical chatbot](https://github.com/user-attachments/assets/7bf25766-3d2e-40c0-934a-a9a274ba2e4b)


Dataset : 

Mle_Screening_Dataset: https://drive.google.com/file/d/1upzfj8bXP012zZsq01jcoeO9NyhmTHnQ/view?usp=drive_link<img width="468" height="38" alt="image" src="https://github.com/user-attachments/assets/35ceb60d-c6da-4366-a46a-f720d3c6bfce" />

 Methodology 

  1. Data Collection And Data Preprocessing 

     Dataset: Got the Mle Screening Dataset from the link given
     
     Data Preprocessing

     Step 1: Data Cleaning and Normalization

   •	 Data Loading: Imported medical Q&A dataset from CSV format using panda
   •	 Missing Value Handling: Removed rows with null or empty values in question and answer column
   •	Text Normalization: Applied lowercase conversion, whitespace trimming, and standardized spacing using regex pattern
   •	Question Mark Standardization: Implemented custom function to normalize question formatting by removing internal question marks and ensuring single question mark at the end

     Step 2: Data Deduplication and Aggregation

      •	Duplicate Removal: Eliminated duplicate question-answer pairs to prevent model overfitting
      •	Question Grouping: Grouped identical questions together using pandas groupby functionality
      •	Answer Consolidation: Combined multiple answers for the same question into comprehensive responses
      •	Data Validation: Verified data integrity and counted final processed records for quality assurance

      Step 3: Database Storage and Dataset Splitting
      
      •	PostgreSQL Integration: Established database connection and created structured tables for efficient data storage
      • Batch Data Loading: Used psycopg2 with execute_values for optimized bulk data insertion into medical_qa table
      •	 Dataset Partitioning: Split processed data into training (80%), validation (10%), and testing (10%) sets using scikit-learn
      •	  Format Standardization: Applied T5-compatible input formatting with "question:" prefix and structured input-output pairs for model training

**2. Model Development and Training**

**Model Architecture Selection : **
•	Implemented combination of BERT-based retrieval system and T5-based generation model using seq2seq architecture
•	Used SentenceTransformer (all-MiniLM-L6-v2) for semantic similarity search and context retrieval from knowledge base
•	Fine-tuned T5-small model for conditional text generation optimized for medical question answering using sequence-to-sequence learning

**Model Training and Validation**
•	Applied sequence-to-sequence learning with teacher forcing mechanism for medical question-answer pair generation
•	 Used cross-entropy loss function for optimizing token-level predictions in answer generation
•	 Saved trained T5 model weights and tokenizer configurations for inference deployment
•	 Validated model with sample medical questions during training to ensure proper contextual answer generation

**3. Implementation & Training Setup**
     
We conducted experiments to train and fine-tune a hybrid model combining BERT retrieval with T5 generation for medical question-answering (QA) tasks. Below are the details of the experimental design and hyperparameters used for our approach:
Hybrid BERT+T5 Model:

**Training Data**: We utilized a processed dataset consisting of medical question-answer pairs from the provided dataset, enhanced with contextual examples through semantic retrieval
Model Architecture: Combined SentenceTransformer (all-MiniLM-L6-v2) for retrieval with T5-small for sequence-to-sequence generation

**Hyperparameters for Training:**
•	Learning Rate: 5e-4
•	Training Epochs: 2
•	Batch Size: 2 (per device for both training and evaluation)
•	Warmup Steps: 10
•	Context Enhancement Threshold: 0.4 (cosine similarity)
•	Maximum Input Length: 200 tokens
•	Maximum Output Length: 50 tokens

**Training Configuration:**
•	Evaluation Strategy: Epoch-based evaluation with best model selection
•	Optimizer: AdamW (default in training framework)
•	Hardware Acceleration: MPS (Metal Performance Shaders) for Apple M2 chip
•	Data Collation: Seq2Seq padding strategy for variable length sequences
•	Context Retrieval: Top-K retrieval (K=1) with similarity-based filtering

**Model Enhancement:**
•	Context Augmentation: Enhanced training samples with retrieved similar examples when similarity > 0.4
•	Sample Limitation: Used 50 training samples and 20 validation samples for efficient training on limited resources.

**4. Web Application Development
Frontend Implementation**
•	Framework Selection: Developed user interface using Streamlit for rapid prototyping and intuitive medical query interaction
•	User Interface Design: Created simple, clean interface with text input field for medical questions and display area for generated answers
•	User Experience Features: Implemented input validation, loading states, and error handling for seamless user interaction
•	Response Display: Formatted model answers with success/error messaging and clear visual feedback for user queries

**Backend API Development**
•	API Framework: Built RESTful API using FastAPI for high-performance backend services and model inference
•	Endpoint Configuration: Created /ask POST endpoint to handle medical question processing and return generated answers
•	Model Integration: Integrated trained BERT+T5 model with backend API for real-time question answering capabilities
•	Request Handling: Implemented Pydantic models for request validation and structured JSON response formatting

**System Integration & Deployment**
•	Client-Server Architecture: Established communication between Streamlit frontend and FastAPI backend via HTTP requests
•	Model Loading: Configured backend to load trained model and knowledge base once during initialization for efficient inference
•	Error Management: Implemented comprehensive error handling for backend connectivity, model inference, and user input validation
•	Local Deployment: Set up local development environment with backend running on port 8000 and frontend interface for testing

**4. Model Evaltuation:**
**Evaluation Metrics Framework**
Word Overlap Score: Measured lexical similarity between generated and expected answers using token-level comparison
Semantic Similarity Score: Applied sentence transformer models to evaluate contextual and semantic alignment between predictions and ground truth
Overall Performance Score: Combined word overlap and semantic similarity metrics to provide comprehensive evaluation assessment.
Top Example Analysis: Identified and analyzed highest-performing examples to understand model strengths and response patterns.

**Performance Results Obtained**
•	Validation Set Performance: Achieved overall score of 0.570 with word overlap of 0.538 and semantic similarity of 0.592.
•	Test Set Performance: Obtained overall score of 0.590 with word overlap of 0.531 and semantic similarity of 0.630.
•	Semantic Understanding: Demonstrated strong semantic comprehension with scores above 0.59 on both validation and test sets.
•	Performance Classification: Model achieved "GOOD" performance rating across both evaluation datasets.

****5.Results ****
**Performance Metrics**
•	Overall Score: 0.59 on test dataset (0.57 on validation)
•	Semantic Similarity: 0.630 (test) / 0.592 (validation)
•	Word Overlap: 0.531 (test) / 0.538 (validation)
•	Performance Rating: "GOOD" classification on both datasets

**Deployment**
**Local Deployment Setup**
•	Backend Service: Deployed FastAPI server on localhost:8000 for handling medical query processing and model inference.
•	Frontend Interface: Launched Streamlit web application providing user-friendly interface for medical question submission.
•	Model Integration: Configured backend to load trained BERT+T5 model and knowledge base during initialization for real-time responses.


**System Architecture**
•	Client-Server Communication: Established HTTP requests between Streamlit frontend and FastAPI backend for seamless user interaction.
•	Hardware Optimization: Configured for Apple M2 chip with MPS acceleration for efficient model inference.
•	Database Connection: Integrated PostgreSQL database for accessing processed medical knowledge base during retrieval operations.

<img width="977" height="613" alt="Screenshot 2025-08-17 at 4 33 09 PM" src="https://github.com/user-attachments/assets/f71017f8-3f14-4d0f-92eb-3a265c6a76d4" />






 
