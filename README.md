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

```
Input tokens:  x1 → x2 → x3 → x4
   x1   → [h1] → 
                ↓
   x2   → [h2] → 
                ↓
   x3   → [h3] → 
                ↓
   x4   → [h4] → Output
```

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

```
Input tokens:  x1    x2    x3    x4

   x1 ──┐
   x2 ──┼──► [Self-Attention] ──► Output1, Output2, Output3, Output4
   x3 ──┤
   x4 ──┘
```

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

**2023 — The Year of Generative AI**
- **ChatGPT (GPT-3.5)** was released and quickly adopted by the public, gaining massive media coverage.  
- Not only GPT-3.5, but several other models also made a significant impact in 2023.
---

# Large Language Models (LLMs)

The term **"large"** is arbitrary — what is considered large today may be small tomorrow.  
Currently, LLMs can range from **1 billion to 60+ billion parameters**.

---

#### Traditional ML vs LLM
- Traditional machine learning: Models are trained for **specific tasks** (e.g., classification).  
- LLMs: Trained to understand and generate natural language, enabling **general-purpose capabilities**.

---

<img width="600" height="161" alt="image" src="https://github.com/user-attachments/assets/709afd8f-c5c8-4fe5-bc9d-c8199ca52da9" />


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

#### LLM Frameworks
1. Open-source LLMs require special backend packages to run locally, unlike closed-source models that you access via API.
2. In 2023, many frameworks emerged, making it overwhelming to choose from hundreds.
3. Instead of covering all, the focus is on building intuition so you can easily learn new frameworks.
4. Backend packages (no GUI) are highlighted since they load and run LLMs efficiently.
5. Examples include llama.cpp, LangChain, and Hugging Face Transformers.

---

#### Generating you first text
An important component of using language models is selecting them. 
The main source for finding and downloading LLMs is the Hugging Face Hub. 
Hugging Face is the organization behind the well-known Transformers package, 
which for years has driven the development of language models in general.

[Hugging Face Hub](https://huggingface.co/)

When you use an LLM, two models are loaded:
1. The generative model itself
2. Its underlying tokenizer

1. The tokenizer is in charge of splitting the input text into tokens before feeding it to the generative model.
2. we use “microsoft/Phi-3-mini-4k-instruct” as the main path to the model.
3. We can use transformers to load both the tokenizer and model.

Note: assume you have an NVIDIA GPU (device_map="cuda") but you can choose a different device instead. (Using Colab)

1. Installing Dependencies
```python
# %%capture
# !pip install transformers>=4.40.1 accelerate>=0.27.2
```

2. The first step is to load our model onto the GPU for faster inference.
Note that we load the model and tokenizer separately (although that isn't always necessary).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
```
Running the code will start downloading the model and depending on your internet connection can take a couple of minutes.

3. Although we now have enough to start generating text, there is a nice trick in transformers that simplifies the process,
namely transformers.pipeline. 
It encapsulates the model, tokenizer, and text generation process into a single function:

```python
from transformers import pipeline

generator = pipeline(
    "text-generation", 
    model="microsoft/Phi-3-mini-4k-instruct"
)
```
4. To generate our first text, let’s instruct the model to tell a joke about chickens.
Our role is that of “user” and model role is of assistant
we use the “content” key to define our prompt:

```python
prompt = "User: Create a funny joke about chickens.\nAssistant:"
response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print(response[0]["generated_text"])
```

```
output:
User: Create a funny joke about chickens.
Assistant: Why did the chicken join a band? Because it wanted to be a drumstick.
```

## Parameters

- **return_full_text**  
  If `True`, the output will include both the input (prompt) and the model’s generated output.  
  If `False`, only the newly generated text is returned.  

- **max_new_tokens**  
  `max_new_tokens=N` → Generate at most **N tokens** after the prompt.  

- **do_sample**  
  Controls whether generation is **deterministic** or **stochastic (randomized)**.  
  1. If `False` → The model always picks the token with the highest probability (**greedy decoding**).  
  2. If `True` → The model samples from the probability distribution (**adds randomness, more diverse outputs**).  

---

### Other Chapters:
1. Chapter 2: [Tokens & Embeddings](https://github.com/Aditya-1998k/Hands-on-Large-Language-Model/blob/main/Tokens%20and%20Embeddings.md)
