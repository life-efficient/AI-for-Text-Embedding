# %%
# TITLE: AI FOR TEXT EMBEDDINGS
from jokes import jokes  # get some data
import string
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# TOKENISATION GET UNIQUE LIST OF WORDS


def joke_tokeniser(corpus):
    tokens = []
    for joke in jokes:
        joke_tokens = []
        for line in joke:
            # CLEAN THIS DATA
            line = line.lower()
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = line.replace('’s', '')
            joke_tokens.extend(line.split())
        tokens.extend(
            joke_tokens
        )
    tokens = set(tokens)
    return tokens


tokens = joke_tokeniser(corpus=jokes)
print(tokens)

# CREATE SOME HANDCRAFTED EMBEDDINGS AND APPLY TO EACH WORD


def embed(word):
    embedding = [
        get_num_vowels(word),
        get_num_consonants(word),
        len(word)
    ]
    return embedding


def get_num_vowels(word):
    return len([char for char in word if char in ("a", "e", "i", "o", "u")])


def get_num_consonants(word):
    return len([char for char in word if char not in ("a", "e", "i", "o", "u")])


# VISUALISE THAT IN 3D SPACE
embeddings = np.array([embed(word) for word in tokens])
writer = SummaryWriter()
writer.add_scalar('test', 100)
print(embeddings)
print(list(tokens))
writer.add_embedding(
    mat=embeddings,
    metadata=list(tokens)
)

# VISUALISER HIGHER DIM EMBEDDINGS
# LOOK INTO MORE ADVANCED EMBEDDINGS BERT
# LEARN OUR OWN EMBEDDINGS FROM SCRATCH

# CREATE VOCAB = MAPPING BETWEEN EACH TOKEN IN YOUR CORPUS AND AN INTEGER INDEX
# [0, 4, -1]

# %%
