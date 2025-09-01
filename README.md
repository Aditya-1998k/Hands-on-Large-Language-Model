# LLM Learning Notes

1. What is Language AI?  
2. What is a Large Language Model?  
3. What are common applications of LLMs?  
4. How can we use them ourselves?  

---

### What is Language AI?  
Artificial Intelligence (AI) = computer systems designed to perform human-level tasks.  

Examples:  
1. Speech recognition (Siri, Alexa)  
2. Language translation (Google Translate)  
3. Visual perception (object detection in images)  

Single Sentence: Intelligent Machine == AI == Intelligent Computer Program  

Language AI = Natural Language Processing (NLP)  

![Ref: O’Reilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/x9w0r7hohsvgubkvtxv3.png)

Language is tricky for computers. Text is unstructured data and loses meaning when converted into 0s and 1s.  

---

### Language as Pack of Words  
First step: Tokenization (splitting text into smaller parts).  

Methods:  
- Split on whitespace  
- Split into individual words  

![Credit: O’Reilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/dspn9e4wx1wb5toppae5.png)

Problem: Some languages (e.g., Mandarin) don’t use whitespace → simple tokenization fails.  

---

### Dense Vector Embeddings  
To preserve meaning, Dense Vector Embeddings are used.  
- Bag of Words = ignores semantics  
- Embeddings = capture meaning of words  

---

### Word2Vec (2013)  
A neural network approach that produces word embeddings by predicting context.  

- Each connection has a weight (model parameter).  
- Learns relationships between words during training.  
- If two words appear often together → embeddings are closer.  

![Credit: O’Reilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/m4p0ds213wzcaynnnya8.png)

Example:  
- baby → high score on newborn, human  
- apple → low score on newborn, human  

![Credit: O’Reilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ovijtkihaiylg605av6x.png)

---

### Why Embeddings Are Helpful  
- Capture semantic similarity between words  
- Go beyond surface-level text meaning  

### Types of Embeddings

- **Sentence embeddings**  
- **Word embeddings**  
- **Document embeddings**

`word2vec` generates embeddings only at the **word level**.

---

## Why Embeddings Are Important
Embeddings are central to many NLP tasks:  
1. Classification  
2. Clustering  
3. Semantic Search  
4. Retrieval-Augmented Generation (RAG)  

---

## Static Nature of Word2Vec
Training with `word2vec` creates **static embeddings**.  
- Example: the word *bank* → always has the same embedding.  
- But *bank* can mean **financial institution** or **riverbank**, depending on context.  

Thus, embeddings need to change with context.  

---

## RNNs and Contextual Embeddings
**RNNs (Recurrent Neural Networks)** are neural networks designed to model sequences.  

They are used for two main tasks:  
1. **Encoding**: representing an input sentence  
2. **Decoding**: generating an output sentence  

RNNs are **autoregressive**:  
- When generating the next word, they rely on previously generated words.  
- Problem: difficult for **long sentences** (information loss).  

---

## Attention Mechanism
Introduced in 2014 to address RNN limitations.  

- Allows the model to **focus on relevant parts** of the input sequence.  
- Amplifies important signals.  
- Selectively determines which words matter most in a sentence.

# Attention is All You Need

In 2017, a new architecture called the **Transformer** was introduced.  
It is based solely on the **attention mechanism** and completely removes recurrence.  

Key benefits:  
- Enables **parallel training**, making it much faster than RNNs.  
- Both encoder and decoder blocks revolve around **attention** rather than RNNs.  

---

## Transformer Architecture

- **Encoder** = Self-Attention + Feedforward Neural Network  
- **Decoder** = Masked Self-Attention + Encoder-Decoder Attention + Feedforward Neural Network  

This design became the foundation of many impactful models such as **BERT** and **GPT-1**.  

---

## Encoder-Only Models
Encoder-only transformers are designed for **understanding tasks**:  
- Learn contextual embeddings for the entire input sequence.  
- Commonly used for **classification**, **clustering**, and **semantic similarity**.  
- Example: **BERT** (Bidirectional Encoder Representations from Transformers).  

---

## Decoder-Only Models
Decoder-only transformers are designed for **generation tasks**:  
- Autoregressive in nature (generate the next token step by step).  
- Used for **text generation**, **summarization**, **chatbots**, etc.  
- Examples: **GPT series**.  

# Generative LLMs

Generative LLMs can be seen as **sequence-to-sequence machines**:  
- They take in some text and attempt to **autocomplete it**.  
- Instead of just completing text, they can be trained to **answer questions** or perform specific tasks.  

---

**Fine-Tuning**
- **Fine-tuning** allows base models to become **instruction-following (instruct)** or **chat models**.  
- These models can take a **user query (prompt)** and generate a more aligned **response**.  

<img width="600" height="186" alt="image" src="https://github.com/user-attachments/assets/c88b503f-667f-4b88-b459-31e91a2a291a" />


---

**Completion Models**
Generative models are also referred to as **completion models**.  
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/1c176546-6fec-4d7c-bc7b-cc7066ca5e59" />


**Example:**  
```
User Query: "Tell me something about llamas"
         ↓
Generative LLM (Task: complete the input)
         ↓
Output: "Llamas are domesticated South American animals related to alpacas..."
```

---

**Context Length & Context Window**
- **Context length**: Maximum number of tokens a model can process.  
- **Context window**: The actual span of tokens (input + output) the model can handle.  
- These are critical to the capability of a generative model.  

<img width="600" height="255" alt="image" src="https://github.com/user-attachments/assets/a1f8bd16-5931-4bf3-9f72-3c5d332907f9" />

---

# Large Language Models (LLMs)

The term **"large"** is arbitrary — what is considered large today may be small tomorrow.  
Currently, LLMs can range from **1 billion to 60+ billion parameters**.

---

#### Traditional ML vs LLM
- Traditional machine learning: Models are trained for **specific tasks** (e.g., classification).  
- LLMs: Trained to understand and generate natural language, enabling **general-purpose capabilities**.

---

### How LLMs are Created

#### 1. Language Modeling (Pre-training)
- The **first step** in building an LLM.  
- Requires the **majority of computation and training time**.  
- The output is often called a **foundation model** or **base model**.  
- These models usually **do not follow instructions** directly.  

#### 2. Fine-Tuning (Post-training)
- The **second step**, where the base model is further trained on **narrower tasks**.  
- Allows the model to **adapt to specific tasks** or exhibit **desired behaviors**.  
- Fine-tuned models can become **instruction-following** or **chat models**.  

---

#### Applications of LLMs
1. **Customer review classification**  
2. **Identifying common issues** in support tickets  
3. **Inspection of documents** (summarization, analysis, compliance)  
4. **LLM-powered chatbots**  
5. **Creative suggestions**, e.g., suggesting a dish to cook from a fridge picture  




---

**2023 — The Year of Generative AI**
- **ChatGPT (GPT-3.5)** was released and quickly adopted by the public, gaining massive media coverage.  
- Not only GPT-3.5, but several other models also made a significant impact in 2023.  
