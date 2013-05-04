from collections import defaultdict
import collections
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk
import pdb
import math



def document_features(document, tagger_output):
    """
    This function takes a document and a tagger_output=[(word,tag)]
    (see functions below), and tells you which words were present as
    'words' (as opposed to 'tags') in tagger_output.

    Parameters
    ----------
    document: string, your document
    tagger_output: list of tuples of the form [(word, tag)]
    
    Returns
    -------
    features : dictionary of tuples ('has(word)': Boolean)

    Notes
    -----
    Use the nltk.word_tokenize() to break up your text into words 
    """
    all_words = set(onlyAlpha(nltk.word_tokenize(document)))
    features = {}
    for word in all_words:
        features[word] = (word in dict(tagger_output)) 

    return features
    

def checkFeatures(document, feature_words):
    """
    This function takes a document and a list of feautures, i.e. words you
    have identitifed as features, and returns a dictionary telling you which
    words are in the document

    Parameters
    ----------
    document: list of strings (words in the text you are analyzing)
    features: list of strings (words in your feature list)

    Returns
    -------
    features: dictionary
        keys are Sting (the words)
        values are Boolean (True if word in feature_words)
    """
    features = {}
    document_words = set(document)
    for word in feature_words:
        features[word] = (word in document_words)

    return features


def onlyAlpha(document):
    """
    Takes a list of strings in your document and gets rid of everything that
    is not alpha, i.e. returns only words

    Parameters
    ----------
    document: list of strings

    Returns
    -------
    words: list of strings
    """
    words = [word for word in document if word.isalpha()]

    return words


def getTopWords(word_list, percent):
    """
    Takes a word list and returns the top percent of freq. of occurence.
    I.e. if percent = 0.3, then return the top 30% of word_list.
    
    Parameters
    ----------
    word_list: list of words
    percent: float in [0,1]

    Returns
    -------
    top_words: list 

    Notes
    -----
    Make sure this returns only alpha character strings, i.e. just words.
    Also, consider using the nltk.FreqDist()
    """
    ###get rid of non alphas in case you have any
    word_list = onlyAlpha(word_list)
    whole_list = nltk.FreqDist(word.lower() for word in word_list)
    cutoff = int(math.ceil(len(whole_list)*percent))
    top_words = whole_list.keys()[:cutoff]

    return top_words


def posTagger(documents, pos_type=None):
    """
    Return all unique part of speech tags in documents.

    Takes a list of strings, i.e. your documents, and tags all the words in
    each string using the nltk.pos_tag().
    In addition if pos_type is not None the function will return only tuples
    (word, tag) tuples where tag is of type pos_type. For example, if
    pos_type = 'NN' we will get back all words tagged with "NN" "NNP" "NNS" etc

    Parameters
    ----------
    documents: list of strings
    pos_type: string

    Returns
    -------
    tagged_words: list of tuples (word, pos)
        One single list no matter how many documents you have.  

    Notes
    -----
    You need to turn each string in your documents list into a list of words
    and you want to return a list of unique (word, tag) tuples. Use the 
    nltk.word_tokenize() to break up your text into words but MAKE SURE you
    return only alpha characters words.  The order of the returned list does
    not matter.
    """
    documents = '\n'.join(documents)
    all_tokens = nltk.word_tokenize(documents)
    tagged_tokens = set(nltk.pos_tag(all_tokens))
    if pos_type:
        tagged_words = [(word, tag) for (word, tag) in tagged_tokens 
                        if not tag.find(pos_type) and word.isalpha()]
    else:
        tagged_words = [(word, tag) for (word, tag) in tagged_tokens
                    if word.isalpha()]

    return tagged_words


def bigramTagger(train_data, docs_to_tag, base_tagger=posTagger, pos_type=None):
    """
    Takes a list of strings, i.e. your documents, trains a bigram tagger using the base_tagger for a first pass, then tags all the words in the documents. In addition if pos_type is not None the function will return only those (word, tag) tuples where tag is of type pos_type. For example, if pos_type = 'NN' we will get back all words tagged with "NN" "NNP" "NNS" etc

    Parameters
    ----------
    train_data: list of tuples (word, tag), for trainging the tagger
    docs_to_tag: list of strings, the documents you want to extract tags from
    pos_type: string

    Returns
    -------
    tagged_words: list of tuples (word, pos)

    Notes
    -----
    You need to turn each string in your documents list into a list of words and you want to return a list of unique (word, tag) tuples. Use the nltk.word_tokenize() to break up your text into words but MAKE SURE you return only alpha characters words. Also, note that nltk.bigramTagger() is touchy and doesn't like [(word,tag)] - you need to make this a list of lists, i.e. [[(word,tag)]]
    """


    return tagged_words

    
    
def significantWords(untagged_docs, min_chisq=5, ratio=0.75):
    """ 
    Use chisq test of bigram contingency table to measure 
    the association of token with its sentiment

    Parameters
    ----------
    untagged_docs: list of tuples (words, tag)
    min_chisq: lower bound of significant
    ratio: pos/neg ratio, used to determine the sentiment of a word

    Returns
    -------
    significant_words: a 3-key-dict of words set

    """ 
    significant_words = collections.defaultdict(set)
    freq_dist = FreqDist()
    label_freq_dist = ConditionalFreqDist()
    stopping_words = set(nltk.corpus.stopwords.words('english'))
    for tokens, label in untagged_docs:
        for token in tokens:
            if token.isalpha() and not (token in stopping_words):
                freq_dist.inc(token)
                label_freq_dist[label].inc(token)
    n_xx = label_freq_dist.N()
    #pdb.set_trace()
    for label in label_freq_dist.conditions():
        for word, n_ii in label_freq_dist[label].iteritems():
            n_xi = label_freq_dist[label].N()
            n_ix = freq_dist[word]
            n_oi = n_xi-n_ii
            n_io = n_ix-n_ii
            n_oo = n_xx-n_oi-n_io-n_ii
            chisq = float(n_xx*(n_ii*n_oo - n_io*n_oi)**2)\
                    /((n_ii+n_io)*(n_ii+n_oi)*(n_oo+n_io)*(n_oo+n_oi))
            if chisq > min_chisq and n_ii>10:
                significant_words['total'] |= set([word])
                if float(n_ii)/n_ix > ratio and (n_ix-n_ii) > 1:
                    significant_words[label] |= set([word])
    return significant_words





    





