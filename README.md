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
