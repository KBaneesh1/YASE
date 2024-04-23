import streamlit as st
from preprocess import CustomTokenizer
from statistics import TfidfRanker, add_page_rank_scores_and_reorder
import pickle
from pseudo_relevance_feedback import CustomPseudoRelevanceFeedback
import webbrowser

N_PAGES = 10000
RESULTS_PER_PAGE = 10
MAX_RESULTS_TO_CONSIDER = 100

with open('url_from_code_dict.pickle', 'rb') as handle:
    url_from_code = pickle.load(handle)
with open('code_from_url_dict.pickle', 'rb') as handle:
    code_from_url = pickle.load(handle)
with open('inverted_index_dict.pickle', 'rb') as handle:
    inverted_index = pickle.load(handle)
with open('doc_lengths_dict.pickle', 'rb') as handle:
    docs_length = pickle.load(handle)
with open('page_ranks_dict.pickle', 'rb') as handle:
    page_ranks = pickle.load(handle)
with open('docs_tokens_dict.pickle', 'rb') as handle:
    docs_tokens = pickle.load(handle)

tokenizer = CustomTokenizer(N_PAGES)
tf_idf_ranker = TfidfRanker(inverted_index, N_PAGES, page_ranks, docs_length, True)

st.title("UIC Web Search Engine")

use_pseudo_relevance_feedback = st.checkbox("Enable Pseudo-Relevance Feedback")

query = st.text_input("Enter your query:")

if query:
    query_tokens = tokenizer.tokenize(query)
    best_ranked = tf_idf_ranker.retrieve_most_relevant(query_tokens)[:MAX_RESULTS_TO_CONSIDER]

    if use_pseudo_relevance_feedback:
        pseudo_relevance_feedback = CustomPseudoRelevanceFeedback(inverted_index, best_ranked, docs_tokens)
        pseudo_relevance_feedback.run_pseudo_relevance()
        query_expansion_tokens = pseudo_relevance_feedback.get_query_expansion_tokens(query_tokens)
        best_ranked_expanded = tf_idf_ranker.retrieve_most_relevant_expanded(query_tokens, query_expansion_tokens)[:MAX_RESULTS_TO_CONSIDER]
        
        st.write("Expanded Query:", ' '.join(query_expansion_tokens))
        st.write("Results:")
        for doc_code, _ in best_ranked_expanded:
            url = url_from_code.get(doc_code, "Unknown URL")
            st.write(f"[{url}]({url})") 
           
    else:
        st.write("Results:")
        for doc_code, _ in best_ranked:
            url = url_from_code.get(doc_code, "Unknown URL")
            st.write(f"[{url}]({url})")  
         
