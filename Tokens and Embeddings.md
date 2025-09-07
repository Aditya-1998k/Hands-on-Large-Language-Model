
# Tokens & Embeddings
Tokens and Embeddings are two central concepts of using LLMs.
They provide clear sense how AI works and how LLM 

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/user-attachments/assets/b000b1a6-1fa3-4aaa-bd70-1f8953eb83b2" alt="image" width="600" height="384">

- Token == Breaking down the sentences
- Embeddings == Converting tokens into numeric representation (vectors)

## LLM Tokenization
Before propmt is presented to the LLM, first it have to be go through the tokenizer that breaks it into the peices.

Example showing the tokenizer of GPT-4 on the OpenAI Platform

&nbsp;&nbsp;&nbsp;&nbsp;<img width="600" height="431" alt="image" src="https://github.com/user-attachments/assets/4cddf1b8-e56b-4e95-a2da-59ec32c35826" />


Tokenization Scheme:
1. Word-level → splits text by spaces (e.g., “playing football” → ["playing", "football"]).
2. Character-level → each character is a token (e.g., “cat” → ["c", "a", "t"]).
3. Subword-level → breaks words into smaller frequent pieces (e.g., “unhappiness” → ["un", "happi", "ness"]).
4. Byte/byte-pair → works at the byte level, ensuring every symbol is covered (used in GPT-2, LLaMA).

Note: Modern LLMs mostly use subword or byte-level tokenization for flexibility and efficiency.

### Comparing Trained LLM Tokenizers
Different Type of tokens:
1. Capitalization.
2. Languages other than English.
3. Emojis.
4. Programming code with keywords and whitespaces often used for indentation (in languages like Python for example).
5. Numbers and digits.
6. Special tokens. These are unique tokens that have a role other than representing text.
They include tokens that indicate the beginning of the text, or the end of the text
(which is the way the model signals to the system that it has completed this generation),
or other functions as we’ll see.

```python
text = """
English and CAPITALIZATION

🎵鸟
show_tokens False None elif == >= else: two tabs:" " Three tabs: "   "

12.0*50=600
"""
```

Method to use Different tokenizers:
```python
colors_list = [
    '102;194;165', '252;141;98', '141;160;203', 
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' + 
            tokenizer.decode(t) + 
            '\x1b[0m', 
            end=' '
        )
```

#### Model : BERT base model (uncased) (2018)
Pretrained model on English language using a masked language modeling (MLM) objective.
It was introduced in this paper and first released in this repository.
This model is uncased: it does not make a difference between english and English.

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. 
This means it was pretrained on the raw texts only, 
with no humans labeling them in any way (which is why it can use lots of publicly available data) 
with an automatic process to generate inputs and labels from those texts.

Tokenization method: WordPiece

Special tokens include:
1. [UNK] → for unknown tokens.
2. [SEP] → to separate two texts (e.g., in cross-encoders).
3. [PAD] → to pad sequences to a fixed length.
4. [CLS] → for classification tasks, and [MASK] → to hide tokens during training.

<img width="829" height="150" alt="image" src="https://github.com/user-attachments/assets/36ad384a-5714-4bf1-9b47-700fab4e4034" />

Key points:
1. Newline breaks are lost, so structure (like chat turns) is hidden.
2. Text is lowercased.
3. Words may split into subtokens (e.g., capital + ##ization), with ## marking continuation.
4. Emojis/Chinese characters become [UNK] (unknown tokens).

Other Models:

<img width="811" height="337" alt="image" src="https://github.com/user-attachments/assets/6d8fe011-0c95-4037-a487-038cbcb9635a" />

### Tokenizer Properties
What determines tokenizers behaviours:
1. Tokenization Method: No. of tokenization method like BPE (byte pair encoding)
2. Tokenizer Parameters: LLM designer needs to make some decision about parameter of tokenizer
   - Vocabulary size (How many tokens to keep in the tokenizer’s vocabulary?) (30K and 50K are often used)
   - Special tokens : What special tokens do we want the model to keep track of?
   - Capitalization : In languages such as English, how do we want to deal with capitalization? Should we convert everything to lowercase?
3. The domain of the data: Even if we select the same method and parameters, tokenizer 
behavior will be different based on the dataset it was trained on (before we even start model training).


## Token Embeddings

Tokenization solved on part of the problem serving language to the language model. In this sense
language is a sequence of token. If we train good enough model on large enough set of tokens, it
starts to capture the complex pattern that appears in its training dataset.

1. If the training data contains a lot of English text, that pattern reveals 
itself as a model capable of representing and generating the English language.
2. If the training data contains factual information (Wikipedia, for example), 
the model would have the ability to generate some factual information (see the following note).

The next piece of the puzzle is **finding the best numerical representation for these tokens** that the 
model can use to calculate and properly model the **patterns in the text**. 
These patterns reveal themselves to us as a model’s coherence in a specific language, 
or capability to code, or any of the **growing list of capabilities** we expect from language models.

Note: A Language Model Holds Embeddings for the Vocabulary of Its Tokenizer

**Why we can't use different tokenizer for pre-trained model?**
After a tokenizer is initialized and trained, it is then used in the training process of its associated language model. 
This is why a pretrained language model is linked with its tokenizer and can’t use a different tokenizer without training.

Before beginning of the training process, these vectors are randomly intialized, but training process assign them
the values that enables the useful behaviour they are trained to perform.

<img width="600" height="321" alt="image" src="https://github.com/user-attachments/assets/4cdb444f-f95f-44dc-942a-a2fd707a4dee" />

#### Contextualized Word Embeddings with Large Language Models
Instead of representing each word or token with a static vector, language models create contextualized word embeddings
that represent a word with different token  based on its context.

What powers AI image generation system like DALL.E, Midjourney and stable diffusion?

<img width="600" height="279" alt="image" src="https://github.com/user-attachments/assets/7805e038-f102-4e70-abf6-3f2a45efb5fc" />

#### Let's Generate
1. Load a tokenizer and model
```python
from transformer import AutoModel, AutoTokenizer

# Load Tokenizer
tokenizer = AutoTokenizer("microsoft/deberta-base")

# Load Model
model = AutoModel("microsoft/deberta-v3-xsmall")
```

- Tokenizer → Converts text into tokens (numbers).
- Model → Neural network that converts tokens into embeddings.

2. Tokenization
```python
tokens = tokenizer("Hello World", return_tensors="pt")
```
- `Hello World` get splitted into tokens
- Returned a pytorch tensors for model processing

3. Model Output
```python
# Process the token
output = model(**tokens)[0]

output.shape
# torch.Size([1, 4, 384])
```
Shape Meaning
- 1 = Batch size (We only gave one sentence)
- 4 = tokens
- 384 = each token is represented by 384-dimensional vectors

4. Inspecting Tokens
```
for token in tokens["input_ids"][0]:
    print(tokenizer.decode(token))

# Output
[CLS]
Hello
world
[SEP]
```

Tokenizer added special tokens
- [CLS] = classification token (at start)
- [SEP] = seperator token (at end)

So "Hello world" → actually became 4 tokens.

5. Embedding Vectors
Model Output
```
tensor([[
[-3.3060, -0.0507, ..., 0.6932],   # [CLS]
[ 0.8918,  0.0740, ..., 0.0751],   # Hello
[ 0.0871,  0.6364, ..., 1.0157],   # world
[-3.1624, -0.1436, ..., 0.7954]    # [SEP]
]])
```

**ASCII Diagram**
```python
Text: "Hello world"
        │
        ▼
+-----------------+
|   Tokenizer     |
+-----------------+
        │
        ▼
Tokens: [CLS], "Hello", "world", [SEP]
        │
        ▼
Token IDs: [101, 8667, 1362, 102]   # Example IDs
        │
        ▼
+-----------------+
|   Embedding     |
| (inside model)  |
+-----------------+
        │
        ▼
Embeddings:
[[-3.3060, -0.0507, ..., 0.6932],   # [CLS]
 [ 0.8918,  0.0740, ..., 0.0751],   # Hello
 [ 0.0871,  0.6364, ..., 1.0157],   # world
 [-3.1624, -0.1436, ..., 0.7954]]   # [SEP]
        │
        ▼
torch.Size([1, 4, 384])
```

### Text Embeddings (Sentence & Documents)
- Token Embeddings : One vector per token
- Text Embeddings : One vector per sentence

Useful when we care about the meaning of the whole text, not just individual words.

**How text embeddings are created?**
1. Method 1 : Average all token embeddings into one vector
2. Method 2 : Use a model trained specifically for text-embeddings (eg: Sentence-transformer model)

Using Sentence Transformer Model

```python
from sentence_transformer import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert sentence into embedding vectors
vector = model.encode("Best Movie Ever!")

print(vector.shape)
# (768,)
```
- The embedding vector has 768 values.
- This single vector represents the meaning of the sentence.

**ASCII Diagram**
```
                 🔹 Token Embeddings                         🔹 Text Embeddings
------------------------------------------------------------------------------------------
Input: "Best movie ever!"                          Input: "Best movie ever!"
        │                                                   │
        ▼                                                   ▼
+---------------------+                             +---------------------------+
|   Tokenizer + LM    |                             |   Text Embedding Model    |
+---------------------+                             | (e.g., all-mpnet-base-v2) |
        │                                                   +-------------------+
        ▼                                                   │
Tokens: ["Best", "movie", "ever", [SEP]]                     ▼
        │                                            Token Embeddings (internal):
        ▼                                            [[-0.12, 0.56, ..., 0.88],   # Best
Token Embeddings:                                    [ 0.34, 0.21, ..., -0.44],  # movie
[[-0.12, 0.56, ..., 0.88],   # Best                 [ 0.77, -0.11, ..., 0.05]]   # ever
 [ 0.34, 0.21, ..., -0.44],  # movie                        │
 [ 0.77, -0.11, ..., 0.05],  # ever                         ▼
 [-0.22, 0.18, ..., 0.14]]   # [SEP]              (Pooling / Special Layer)
        │                                                   │
        ▼                                                   ▼
Shape: (1, 4, 384)                                  Text Embedding:
                                                    [-0.231, 0.884, -0.114, ..., 0.672]
                                                            Shape: (768,)

```

##### Application of Text Embeddings
- Semantic Search - Find similar sentence/Documents
- Categorization/Clustering - Group similar text together
- RAG (Retrieval Augmented Generation) - Find relevant documents to feed into LLMs
- Recommendation system - Suggest similar items based on text

### Word2Vec & Contrastive Training
Word2vec learns word embeddings by predicting wheather two words appear in the same context.
Trained using sliding window on the text to generate the word pairs.

**Skip-Gram**
The skip-gram model is a word embedding technique used in natural language processing to predict the surrounding words given a target word.

1. Central word → paired with its neighbors.
```
Eg:
If window size == 2:
"not make a machine"
Center = make
Pairs = (make, not), (make, a), (make, machine)
```
2. Negative Sampling
Add random words pairs - label as not neibours
Prevent the model from cheating by always predicting 1 (Neighbour)
Inspired by Noise contrastive Estimation.

Noise contrastive estimation is a technique used in natural language processing to learn 
high-quality vector representations by differentiating data from noise.

3. Training process
Initialized embeddings randomly (vocab size * embedding dimension).
```
For each pair: 
input: Two word embeddings
Output: 1 if neighbour else 0

Grandually update embeddings so that :
- similar words - close vectors
- random words - distant vectors
```

**ASCII Diagram**
```
Text: "Thou shalt not make a machine in the likeness of a human mind"

Sliding Window (size=2)
       Center word = "make"
       Neighbors = ["not", "a", "machine"]

Training Pairs:
  Positive (Skip-Gram):
    (make, not) → 1
    (make, a)   → 1
    (make, machine) → 1

  Negative (Random Sampling):
    (make, banana) → 0
    (make, elephant) → 0

+-------------------------------+
|   Neural Network Classifier   |
|   Input: (word1, word2)       |
|   Output: 1 (neighbor) / 0    |
+-------------------------------+
        │
        ▼
Updates Embedding Matrix:
[vocab_size × embedding_dim]

Final Result:
- "make" and "machine" embeddings → close
- "make" and "banana" embeddings → far apart
```

### Recommending songs by Embeddings
We will use word2vec algorithm to embed songs using human made music playlist.
- Treat each songs as a word or token
- Treat each playlist as a sentence

Then Embedding recommend each song which appears together in playlist.

<img width="600" height="127" alt="image" src="https://github.com/user-attachments/assets/d2fba05f-4543-4ec7-ac18-ef39df165f33" />

- About Dataset : [Link](https://www.cs.cornell.edu/~shuochen/lme/data_page.html)
- Colab: [Colab Link](https://colab.research.google.com/drive/1LEuBkO416Ozq5AX2zCQX8c5waH4hNmu7#scrollTo=anBKHSPCUKLk)

Training a song Embedding model
1. Loading dataset and song metadata
```python
import pandas as pd
from urllib import request

# Get the playlist datafile
data = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt")

# Parse the dataset files and Skip the first two lines as they contains only metadata
lines = data.read().decode("utf-8").split("\n")[2:]

# Remove playlist with one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load Song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split("\n")
songs = [s.rstrip().split("\t") for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ["id", "title", "artist"])
songs_df = songs_df.set_index("id")
```
Will create Embeddings on playlist which will have song_ids and will use `songs_df` to get the song details.

```python
print( 'Playlist #1:\n ', playlists[0], '\n')
print( 'Playlist #2:\n ', playlists[1])
```
Output
```
Playlist #1: ['0', '1', '2', '3', '4', '5', ..., '43']
Playlist #2: ['78', '79', '80', '3', '62', ..., '210']
```
```
!pip install gensim
```

Let's train the Model
```python
from gensim.models import word2vec

# Train our Word2Vec model
model = Word2Vec(
    playlists,         # playlists = list of lists of song_ids
    vector_size=32,    # size of embedding (dimension of vector space)
    window=20,         # how many "neighbors" to consider on either side
    negative=50,       # negative sampling
    min_count=1,       # include all songs, even rare ones
    workers=4          # parallelism
)
```
That takes a minute or two to train and results in embeddings being calculated for each song that we have.
```python
song_id = 2172
# Ask the model for songs similar to song #2172
model.wv.most_similar(positive=str(song_id))
```
Output:
```
[('2976', 0.9977465271949768),
 ('3167', 0.9977430701255798),
 ('3094', 0.9975950717926025),
 ('2640', 0.9966474175453186),
 ('2849', 0.9963167905807495)]
```
```python
print(songs_df.iloc[2172])
```
```
title Fade To Black
artist Metallica
Name: 2172 , dtype: object
```

Method to get the song recommendation
```python
import numpy as np

def print_recommendations(song_id):
    similar_songs = np.array(model.wv.most_similar(positive=str(song_id),topn=5))[:,0]
    return  songs_df.iloc[similar_songs]

# Extract recommendations
print_recommendations(2172)
```
```
id	    Title	             artist
--------------------------------------
11473	Little Guitars	     Van Halen
3167	Unchained	         Van Halen
5586	The Last in Lin      Dio
5634	Mr. Brownstone	     Guns N’ Roses
3094	Breaking the Law	 Judas Priest
```


