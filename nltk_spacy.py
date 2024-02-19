import spacy

nlp = spacy.load('en_core_web_sm')
stop_words_spacy = nlp.Defaults.stop_words

from nltk.corpus import stopwords
stop_words_nltk = set(stopwords.words('english'))

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')


# ==========================================================================
# NLTK Data Processing

def nltk_keywords(data):
    from nltk import word_tokenize
    tokens = word_tokenize(data)

    from nltk import pos_tag
    pos_tagged_tokens = pos_tag(tokens)
    keywords = [str(t[0]) for t in pos_tagged_tokens if t[1] in ['NNP', 'NN']]

    keywords = [w for w in keywords if w not in stop_words_nltk]
    keywords = sorted(list(set(x for x in keywords)))

    return keywords


# ==========================================================================
# Spacy Data Processing

def spacy_keywords(data):
    tokens = nlp(data)

    pos_tagged_tokens = [(tok, tok.tag_) for tok in tokens]
    keywords = [str(t[0]) for t in pos_tagged_tokens if t[1] in ['NNP', 'NN']]

    keywords = [w for w in keywords if w not in stop_words_spacy]

    keywords = sorted(list(set(x for x in keywords)))
    return keywords
