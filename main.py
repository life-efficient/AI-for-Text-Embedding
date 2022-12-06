# %%
from utils import visualise_embeddings
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece, WordLevel
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordLevelTrainer
from transformers import BertModel, BertTokenizer
import pandas as pd
from word_types.prepositions import prepositions
# %% CUSTOM TOKENISER
# train_corpus = "pythonwiki.txt"

# def get_tokenizer():
#     tokenizer = Tokenizer(WordLevel())
#     tokenizer.pre_tokenizer = Whitespace()
#     tokenizer.normalizer = normalizers.Sequence([
#         normalizers.Lowercase()
#     ])
#     trainer = WordLevelTrainer()
#     tokenizer.train([train_corpus], trainer)
#     return tokenizer

# tokenizer = get_tokenizer()

# tokenizer.get_vocab()
# for idx in range(len(tokenizer.get_vocab())):
#     print(tokenizer.id_to_token(idx))
# output = tokenizer.encode("rough This parts")
# print(output.tokens)
# print(output.ids)

# %% GET BERT
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)

# EXAMPLE TOKENISATION
sentence = "Now I want to know what does this vector refers to in dictionary"
tokens = bert_tokenizer.encode(sentence)
# print(bert_tokenizer.ids_to_tokens)
# TODO DOES A ROUND TRIP - ENCODING AND DECODING RESULT IN THE SAME THING?


# %% GET EMBEDDINGS FROM BERT MODEL

n_embeddings = 30000

embedding_matrix = model.embeddings.word_embeddings.weight.detach()

embedding_matrix = embedding_matrix[:n_embeddings]
# print(embedding_matrix)
print("Embedding shape:", embedding_matrix.shape)


def create_embedding_labels():
    # ADD NEW COLS
    label_functions = {
        "Length": lambda word: len(word),
        "# vowels": lambda word: len([char for char in word if char in "aeiou"]),
        "is number": lambda word: word.isdigit(),
        "is preposition": lambda word: word in prepositions
    }
    labels = [
        [
            word,
            *[label_function(word) for label_function in label_functions.values()]
        ]
        for word in list(bert_tokenizer.ids_to_tokens.values())[:n_embeddings]
    ]

    label_names = ["Word", *list(label_functions.keys())]

    # SAVE LABELS TSV
    # labels.to_csv("metadata.tsv", sep="\t") # this is the format that tensorboard expects
    return labels, label_names


labels, label_names = create_embedding_labels()


# LABEL BY LENGTH
visualise_embeddings(
    embedding_matrix,
    labels=labels,
    label_names=label_names
)

# %%
