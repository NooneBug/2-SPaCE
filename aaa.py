# %%
from embedding_managers.glove_word_embedding import glove_word_embedding

a = glove_word_embedding()
a.load_from_file('resources/word_embeddings/toy_glove.txt')

# %%
