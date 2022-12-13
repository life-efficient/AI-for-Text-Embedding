# %%
from utils import visualise_embeddings
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece, WordLevel
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordLevelTrainer
from transformers import BertModel, BertTokenizer
import pandas as pd
from word_types.prepositions import prepositions
import torch
import torchmetrics
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

# %%
model.modules
# %% GET EMBEDDINGS FROM BERT MODEL

n_embeddings = 30000

embedding_matrix = model.embeddings.word_embeddings.weight.detach()

embedding_matrix = embedding_matrix[:n_embeddings]
# print(embedding_matrix)
print("Embedding shape:", embedding_matrix.shape)

# # %%


# def myfunc(ag1, arg2, arg3):
#     pass


# a = print
# a("hello world")

# myfunc(1, 2, 3)
# myvar = [1, 2, 3]
# myfunc(*myvar)


# myfunc(1, 2, 3, myarg="hello")
# myfunc(myotherarg=0, myarg="hello")
# %%


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
# visualise_embeddings(
#     embedding_matrix,
#     labels=labels,
#     label_names=label_names
# )

# %%


def get_word_embedding(word):

    bert_tokenizer.tokens_to_ids = {
        token: id for id, token in bert_tokenizer.ids_to_tokens.items()}

    token_id = bert_tokenizer.tokens_to_ids[word]
    embedding = embedding_matrix[token_id]
    return embedding


def analogy_solver(a, b, c, embedding_matrix, labels):
    """
    Solves A is to B what C is to D, given, A, B & C, returning D

    """

    # GET EMBEDDINGS FOR KNOWN WORDS
    a_embedding = get_word_embedding(a)
    b_embedding = get_word_embedding(b)

    # GET TRANSFORMATION APPLIED
    transformation_vector = b_embedding - a_embedding

    # plot the embedding of the interpolation

    c_embedding = get_word_embedding(c)
    print(c_embedding.shape)
    d_embedding = c_embedding + transformation_vector
    print(d_embedding.shape)
    nearest_tokens = get_token_from_embedding(d_embedding)
    for d in nearest_tokens:
        print(f"{a} is to {b} as {c} is to {d}")


def get_token_from_embedding(embedding, n=20):
    # cosine similarity from d_embedding to embedding of all words
    similarity = torchmetrics.functional.pairwise_cosine_similarity(
        embedding.unsqueeze(0), embedding_matrix).squeeze()
    similarity_idx = torch.argsort(similarity, dim=0)
    similarity_idx = similarity_idx[:n]
    return [list(bert_tokenizer.ids_to_tokens.values())[idx] for idx in similarity_idx]


def visualise():
    # VISUALISE
    embedding_matrix = torch.cat(
        (
            embedding_matrix,
            transformation_vector.unsqueeze(0),
            # d_embedding.unsqueeze(0)
        )
    )

    labels = [
        *labels,
        ["INTERPOLATION_VECTOR", 0, 0, 0, 0],
        # ["D_EMBEDDING", 0, 0, 0, 0]
    ]
    visualise_embeddings(
        embedding_matrix,
        labels=labels,
        label_names=label_names
    )


analogy_solver("man", "woman", "king", embedding_matrix, labels)

# %%
