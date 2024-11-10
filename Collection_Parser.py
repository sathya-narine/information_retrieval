#This script is used to parse the collections into dictionary
import numpy as np
import random
import nltk

# Download the 'stopwords' dataset
nltk.download('stopwords')
# Now you can instantiate the stop words
stemmer = nltk.stem.PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

### Processing DOCUMENTS
def parse_collection():
    doc_set = {}
    doc_id = ""
    doc_text = ""
    with open('./Collections/CISI.ALL') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    doc_count = 0
    for l in lines:
        if l.startswith(".I"):
            doc_id = int(l.split(" ")[1].strip())-1
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.
    
    ### Processing QUERIES
    with open('./Collections/CISI.QRY') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    
    qry_set = {}
    qry_id = ""
    for l in lines:
        if l.startswith(".I"):
            qry_id = int(l.split(" ")[1].strip()) -1
        elif l.startswith(".W"):
            qry_set[qry_id] = l.strip()[3:]
            qry_id = ""
    
    ### Processing QRELS
    rel_set = {}
    with open('./Collections/CISI.REL') as f:
        for l in f.readlines():
            qry_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]) -1
            doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])-1
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)
    
    ## Here we check some statistics and info of CISI dataset
    
    print('Read %s documents, %s queries and %s mappings from CISI dataset' %
        (len(doc_set), len(qry_set), len(rel_set)))
    
    number_of_rel_docs = [len(value) for key, value in rel_set.items()]
    print('Average %.2f and %d min number of relevant documents by query ' %
        (np.mean(number_of_rel_docs), np.min(number_of_rel_docs)))
    
    print('Queries without relevant documents: ',
        np.setdiff1d(list(qry_set.keys()),list(rel_set.keys())))
        
    return(doc_set,qry_set,rel_set)
   
def preprocess_string(txt, remove_stop=True, do_stem=True, to_lower=True):
    """
    Return a preprocessed tokenized text.

    Args:
        txt (str): original text to process
        remove_stop (boolean): to remove or not stop words (common words)
        do_stem (boolean): to do or not stemming (suffixes and prefixes removal)
        to_lower (boolean): remove or not capital letters.

    Returns:
        Return a preprocessed tokenized text.
    """
    if to_lower:
        txt = txt.lower()
    tokens = nltk.tokenize.word_tokenize(txt)

    if remove_stop:
        tokens = [tk for tk in tokens if tk not in stop_words]
    if do_stem:
        tokens = [stemmer.stem(tk) for tk in tokens]
    return tokens
    
def preprocess_string_lower(txt, remove_stop=True, do_stem=True, to_lower=True):
    if to_lower:
        txt = txt.lower()
    tokens = nltk.tokenize.word_tokenize(txt)
    if remove_stop:
        tokens = [tk for tk in tokens if tk not in stop_words]
    if do_stem:
        tokens = [stemmer.stem(tk) for tk in tokens]
    return ' '.join(tokens)
    
if __name__ == '__main__':
    parse_collection()