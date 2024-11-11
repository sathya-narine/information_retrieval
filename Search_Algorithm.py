from Collection_Parser import parse_collection, preprocess_string, preprocess_string_lower
from rank_bm25 import BM25Okapi
from Evaluation import mean_reciprocal_rank, mean_average_precision, precision_at_k, recall_at_k, f1_score_at_k
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import wordnet

def expand_query(query_tokens):
    expanded_query = set(query_tokens)
    for token in query_tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                expanded_query.add(lemma.name())
    return list(expanded_query)

def results_from_query_new(qry_id, bm25, qry_set, rel_set, remove_stop, do_stem, to_lower):
    query = qry_set[qry_id]
    rel_docs = []
    if qry_id in rel_set:
        rel_docs = rel_set[qry_id]
    tokenized_query = preprocess_string(query, remove_stop, do_stem, to_lower)
    scores = bm25.get_scores(tokenized_query)
    most_relevant_documents = np.argsort(-scores)
    masked_relevance_results = np.zeros(most_relevant_documents.shape)
    masked_relevance_results[rel_docs] = 1
    sorted_masked_relevance_results = np.take(masked_relevance_results, most_relevant_documents)
    return sorted_masked_relevance_results

def bm25_search(tokenized_corpus):
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25
    
def search_with_tfidf(query, vectorizer, tfidf_matrix, qry_id, rel_set):
    query_string = ' '.join(query)
    query_vector = vectorizer.transform([query_string])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_relevant_documents = np.argsort(-cosine_similarities)
    masked_relevance_results = np.zeros(most_relevant_documents.shape)
    rel_docs = rel_set[qry_id] if qry_id in rel_set else []
    for i, doc_id in enumerate(most_relevant_documents):
        if doc_id + 1 in rel_docs:
            masked_relevance_results[i] = 1
    return most_relevant_documents, cosine_similarities, masked_relevance_results

def search_engine(algorithm=1, query=None, remove_stop=True, do_stem=True, to_lower=True, expand_query_flag=False):
    doc_set, qry_set, rel_set = parse_collection()
    corpus = list(doc_set.values())
    tokenized_corpus = [preprocess_string(doc, remove_stop, do_stem, to_lower) for doc in corpus]
    results = []
    
    if algorithm == 1:
        bm25 = bm25_search(tokenized_corpus)
        
        if query:
            tokenized_query = preprocess_string(query, remove_stop, do_stem, to_lower)
            print('original query:',tokenized_query)
            if expand_query_flag:
                tokenized_query = expand_query(tokenized_query)
                print('expanded query:',tokenized_query)
            scores = bm25.get_scores(tokenized_query)
            print('--->',scores)
            most_relevant_documents = np.argsort(-scores)
            print('--->',most_relevant_documents)
            for doc_id in most_relevant_documents:
                #doc_id_str = str(doc_id + 1)  # Assuming doc_id starts from 0 and doc_set keys start from 1
                if doc_id in doc_set:
                    doc = doc_set[doc_id]
                    results.append((str(doc_id), doc_set[doc_id]))
        else:
            rr_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            k = 10
            for qry_id in qry_set.keys():
                sorted_masked_relevance_results = results_from_query_new(qry_id, bm25, qry_set, rel_set, remove_stop, do_stem, to_lower)
                rr_list.append(sorted_masked_relevance_results)
                precision_list.append(precision_at_k(sorted_masked_relevance_results, k))
                recall_list.append(recall_at_k(sorted_masked_relevance_results, k, len(rel_set[qry_id]) if qry_id in rel_set else 0))
                f1_list.append(f1_score_at_k(sorted_masked_relevance_results, k, len(rel_set[qry_id]) if qry_id in rel_set else 0))
            
            results = [results_from_query_new(qry_id, bm25, qry_set, rel_set, remove_stop, do_stem, to_lower) for qry_id in list(qry_set.keys())]
            print('BM25')
            print('---------')
            print('MRR@10 %.4f' % mean_reciprocal_rank(results))
            print("MAP: %.4f" % mean_average_precision(rr_list))
            print("Recall@10: %.4f" % np.mean(recall_list))
            print("F1@10: %.4f" % np.mean(f1_list))
            
    elif algorithm == 2:
        corpus = [preprocess_string_lower(doc) for doc in doc_set.values()]
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.85, min_df=2, use_idf=True, smooth_idf=True, sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        if query:
            tokenized_query = preprocess_string(query, remove_stop, do_stem, to_lower)
            print('original query:',tokenized_query)
            if expand_query_flag:
                tokenized_query = expand_query(tokenized_query)
                print('expanded query:',tokenized_query)
            most_relevant_documents, cosine_similarities, masked_relevance_results = search_with_tfidf(tokenized_query, vectorizer, tfidf_matrix, None, rel_set)
            print('relevant documents',most_relevant_documents)
            for doc_id in most_relevant_documents:
                #doc_id_str = str(doc_id + 1)  # Assuming doc_id starts from 0 and doc_set keys start from 1
                if doc_id in doc_set:
                    doc = doc_set[doc_id]
                    results.append((str(doc_id), doc_set[doc_id]))
        else:
            rr_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            
            k = 10
            
            for qry_id in qry_set.keys():
                query = preprocess_string(qry_set[qry_id])
                most_relevant_documents, cosine_similarities, masked_relevance_results = search_with_tfidf(query, vectorizer, tfidf_matrix, qry_id, rel_set)
                
                rr_list.append(masked_relevance_results)
                precision_list.append(precision_at_k(masked_relevance_results, k))
                recall_list.append(recall_at_k(masked_relevance_results, k, len(rel_set[qry_id]) if qry_id in rel_set else 0))
                f1_list.append(f1_score_at_k(masked_relevance_results, k, len(rel_set[qry_id]) if qry_id in rel_set else 0))
            
            mrr = mean_reciprocal_rank(rr_list)
            mean_precision = np.mean(precision_list)
            mean_recall = np.mean(recall_list)
            mean_f1 = np.mean(f1_list)
            
            print("TF-IDF")
            print("-------")
            print("MRR: %.4f" % mrr)
            print("Precision@10: %.4f" % mean_precision)
            print("Recall@10: %.4f" % mean_recall)
            print("F1@10: %.4f" % mean_f1)
            
    return results[:10]

if __name__ == '__main__':
    #  Query Example
    #query = "How much do information retrieval"
    while True:
        print("#--------------Simple Information Retrieval---------------------#")
        print("Dataset : CISI")
        usr_opt = int(input("Select your options\n1. Evaluate the System\n2. Search Query\n3. Exit\n"))

        if usr_opt == 1:
            remove_stop = input("Remove stopwords (yes/no): ").lower() == 'yes'
            do_stem = input("Perform stemming (yes/no): ").lower() == 'yes'
            to_lower = True
            algorithm = int(input("Choice of algorithm\n1. BM25\n2. TF-IDF\n"))
            search_engine(algorithm=algorithm, remove_stop=remove_stop, do_stem=do_stem, to_lower=to_lower)
        
        elif usr_opt == 2:
            query = input("Enter your query: ")
            expand_query_flag = input("Expand query (yes/no): ").lower() == 'yes'
            algorithm = int(input("Choice of algorithm\n1. BM25\n2. TF-IDF\n"))
            results = search_engine(algorithm=algorithm, query=query, expand_query_flag=expand_query_flag)
            
            print("Search Results")
            for res in results:
                print(res)
        
        elif usr_opt == 3:
            break
        
        else:
            print("Invalid option. Please try again.")