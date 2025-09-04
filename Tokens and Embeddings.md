
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


