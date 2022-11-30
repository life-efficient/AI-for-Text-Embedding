# %%
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece, WordLevel
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import WordLevelTrainer
from transformers import BertModel, BertTokenizer

# %%
train_corpus = "pythonwiki.txt"


def get_tokenizer():
    tokenizer = Tokenizer(WordLevel())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Lowercase()
    ])
    trainer = WordLevelTrainer()
    tokenizer.train([train_corpus], trainer)
    return tokenizer

# BERT


tokenizer = get_tokenizer()

# %%

tokenizer.get_vocab()
for idx in range(len(tokenizer.get_vocab())):
    print(tokenizer.id_to_token(idx))
output = tokenizer.encode("rough This parts")
print(output.tokens)
# print(output.ids)
# %%
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)


sentence = "Now I want to know what does this vector refers to in dictionary"

bert_tokenizer = BertTokenizer.from_pretrained(model_name)
tokens = bert_tokenizer.encode(sentence)
print(tokens)


# var = 10

# class MyClass:
#     def my_method(self):
#         pass

#     @staticmethod
#     def my_static_method():
#         pass

#     @classmethod
#     def class_method(cls):
#         pass


# myclass = MyClass()
# myclass.my_method()

# MyClass.my_static_method()
# myclass.my_static_method()

# MyClass.class_method()
# myclass.class_method()

# %%
