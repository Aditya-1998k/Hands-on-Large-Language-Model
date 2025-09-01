

1. What is Language AI?
2. What is a Large Language Model?
3. What are common applications of LLMs?
4. How can we use them ourselves?

🧠**What is Language AI?**

Artificial Intelligence (AI) is a computer system designed to perform tasks that usually require human intelligence.

Examples include:
1. 🎤 Speech recognition (e.g., Siri, Alexa)
2. 🌍 Language translation (e.g., Google Translate)
3. 👀 Visual perception (e.g., object detection in images)

👉 Single Sentence: **Intelligent Machine == AI == Intelligent Computer Program**  

**Language AI = Natural Language Processing (NLP)**  

**Example of Language AI**

![Ref: Oreilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/x9w0r7hohsvgubkvtxv3.png)

Language is a tricky concept for computers. Text is **unstructured data** and loses meaning when represented in 0s and 1s.  

**Language as Pack of Words** 

The first step in processing text is to split a sentence into words or subwords (**tokenization**).  
Some common tokenization methods:  
- Splitting on whitespace  
- Creating individual words 


![Credit: Oreilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/dspn9e4wx1wb5toppae5.png)
 
However, these methods have disadvantages. For example, some languages don’t use whitespace, making simple tokenization difficult. i.e Mandarin

**Dense Vector Embeddings**  

To solve the problem of lost meaning in tokenization, **Dense Vector Embeddings** came to the rescue.  
Traditional *Bag of Words* approaches ignore the **semantic nature** or meaning of the text.  

**Word2Vec** was one of the first major steps in solving this issue. It uses **Neural Networks** consisting of interconnected nodes that process information.  

- Each connection has a certain **weight**, depending on the input.  
- These weights are the **parameters of the model**.  

Using this neural network, Word2Vec generates **word embeddings** by predicting which word is likely to appear next in a sentence.


![Credit: Oreilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/m4p0ds213wzcaynnnya8.png)

During training, Word2Vec learns the **relationships between words** and distills them into embeddings.  
If two words tend to be neighbors in context, their embeddings will be closer.  

**Example:**
- **baby** → high score on properties of *newborn*, *human*  
- **apple** → low score on properties of *newborn*, *human*  


![Credit: Oreilly](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ovijtkihaiylg605av6x.png)

**Why Embeddings Are Helpful ** 
Embeddings allow us to measure the **semantic similarity** between words. They capture meaning beyond just surface-level text.  


Will update soon.... 
