from nltk.tokenize import RegexpTokenizer
import numpy as np

special_tokens=set([])
missing_stop_words = set(['of', 'a', 'and', 'to'])
words_tokenizer = None

def get_words_tokenizer():
    global words_tokenizer

    if words_tokenizer:
        return words_tokenizer

    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer

def extract_sentence_words(sentence, remove_missing_emb_words = False,remove_special_tokens = False):
    if (remove_special_tokens):
        for token in special_tokens:
            # Can't do on sentence words because tokenizer delete '***' of tokens.
            sentence = sentence.replace(token, "")
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words

def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            return model[word].reshape(1, 300)
        else:
            #print ('Word missing w2v: ' + word)
            return model['UNK'].reshape(1, 300)