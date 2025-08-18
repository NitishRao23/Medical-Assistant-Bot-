# Medical Assistant Bot

An intelligent medical question-answering system that combines BERT-based retrieval with T5-based generation to provide comprehensive answers to medical queries.

## Project Overview

The Medical Assistant Bot is an advanced question-answering system designed to understand and respond to user queries about medical diseases and health topics. Using a hybrid approach that combines BERT-based retrieval for finding relevant medical information and T5-based generation for creating comprehensive answers, this system provides accurate, contextually relevant medical information through a user-friendly web interface.

### Key Features
- **Hybrid AI Architecture**: BERT + T5 models for superior understanding and generation
- **Real-time Processing**: FastAPI backend for high-performance inference
- **Intuitive Interface**: Streamlit frontend for seamless user experience
- **Comprehensive Evaluation**: Multiple metrics for performance assessment
- **Semantic Search**: Advanced retrieval from curated medical knowledge base

## Problems Addressed

### Current Challenges
- Users struggle to find quick, reliable answers to basic medical questions

### Solutions
- **Immediate Responses**: Trained NLP models provide instant, accurate medical answers
- **Contextual Understanding**: Hybrid BERT+T5 architecture for complex query comprehension
- **Reliable Information**: Consistent answers based on validated medical datasets
- **Semantic Search**: Advanced retrieval from comprehensive knowledge base

## Architecture

![Medical chatbot](https://github.com/user-attachments/assets/7bf25766-3d2e-40c0-934a-a9a274ba2e4b)

### Technical Approach
- **Advanced NLP Implementation**: State-of-the-art BERT + T5 models
- **Intelligent Knowledge Base**: Semantic search with continuously accessible medical data
- **Hybrid Model Architecture**: Retrieval-augmented generation approach
- **Scalable Web Application**: FastAPI backend + Streamlit frontend
- **User-Centric Design**: Intuitive interface for enhanced accessibility

## Dataset

**Source**: [MLE Screening Dataset](https://drive.google.com/file/d/1upzfj8bXP012zZsq01jcoeO9NyhmTHnQ/view?usp=drive_link)

The dataset contains medical question-answer pairs used for training and evaluation.

## Methodology

### 1. Data Collection and Preprocessing

#### Data Cleaning and Normalization
- **Data Loading**: Imported medical Q&A dataset from CSV using pandas
- **Missing Value Handling**: Removed rows with null/empty values
- **Text Normalization**: Applied lowercase conversion, whitespace trimming
- **Question Mark Standardization**: Normalized question formatting

#### Data Deduplication and Aggregation
- **Duplicate Removal**: Eliminated duplicate Q&A pairs
- **Question Grouping**: Grouped identical questions using pandas groupby
- **Answer Consolidation**: Combined multiple answers into comprehensive responses
- **Data Validation**: Verified data integrity and quality assurance

#### Database Storage and Dataset Splitting
- **PostgreSQL Integration**: Structured database storage
- **Batch Data Loading**: Optimized bulk insertion with psycopg2
- **Dataset Partitioning**: Train (80%) / Validation (10%) / Test (10%)
- **Format Standardization**: T5-compatible input formatting

### 2. Model Development and Training

#### Model Architecture Selection
- **Retrieval System**: SentenceTransformer (all-MiniLM-L6-v2) for semantic similarity
- **Generation Model**: Fine-tuned T5-small for conditional text generation
- **Hybrid Approach**: Combined BERT retrieval with T5 generation

#### Training Configuration
```python
# Hyperparameters
Learning Rate: 5e-4
Training Epochs: 2
Batch Size: 2
Warmup Steps: 10
Context Enhancement Threshold: 0.4
Max Input Length: 200 tokens
Max Output Length: 50 tokens
```

#### Model Enhancement
- **Context Augmentation**: Enhanced training with retrieved similar examples
- **Sample Optimization**: 50 training samples, 20 validation samples
- **Hardware Acceleration**: MPS for Apple M2 chip optimization

### 3. Web Application Development

#### Frontend Implementation (Streamlit)
- Clean, intuitive interface for medical query input
- Real-time response display with loading states
- Input validation and error handling
- Success/error messaging with visual feedback

#### Backend API Development (FastAPI)
- RESTful API with `/ask` POST endpoint
- Pydantic models for request validation
- Model integration for real-time inference
- Structured JSON response formatting

#### System Integration
- Client-server architecture via HTTP requests
- Efficient model loading during initialization
- Comprehensive error management
- Local deployment on port 8000

### 4. Model Evaluation

#### Evaluation Metrics
- **Word Overlap Score**: Lexical similarity measurement
- **Semantic Similarity Score**: Contextual alignment evaluation
- **Overall Performance Score**: Combined comprehensive assessment

#### Performance Results
- **Test Set Performance**: Overall Score 0.590
- **Validation Set Performance**: Overall Score 0.570
- **Semantic Understanding**: Strong scores above 0.59
- **Classification**: "GOOD" performance rating

## 5. Results

### Performance Metrics
| Metric | Test Set | Validation Set |
|--------|----------|---------------|
| Overall Score | 0.590 | 0.570 |
| Semantic Similarity | 0.630 | 0.592 |
| Word Overlap | 0.531 | 0.538 |
| Performance Rating | **GOOD** | **GOOD** |

## 6. Deployment

### Local Deployment Setup
1. **Backend Service**: FastAPI server on `localhost:8000`
2. **Frontend Interface**: Streamlit web application
3. **Model Integration**: Pre-loaded BERT+T5 model for real-time inference
4. **Database Connection**: PostgreSQL for knowledge base access

### System Requirements
- Python 3.8+
- PostgreSQL database

## Screenshots

![Medical Assistant Bot Interface](https://github.com/user-attachments/assets/f71017f8-3f14-4d0f-92eb-3a265c6a76d4)

## Usage

1. Start the FastAPI backend server
2. Launch the Streamlit frontend application
3. Enter your medical question in the text input field
4. Receive comprehensive, AI-generated medical information


## Acknowledgments

- Medical dataset providers
- Open-source NLP model contributors
- FastAPI and Streamlit communities

