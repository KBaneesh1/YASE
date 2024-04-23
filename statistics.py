import math
from collections import Counter
import operator

E_CONST = 0.1
PAGE_RANK_MULTIPLIER = 20

def rank_docs(similarities):
    return sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)


def add_page_rank_scores_and_reorder(best_ranked, page_ranks):
    best_dict = dict(best_ranked)
    for doc_code in best_dict:
        best_dict[doc_code] = best_dict[doc_code] + page_ranks[doc_code] * PAGE_RANK_MULTIPLIER
    return rank_docs(best_dict)


class TfidfRanker:

    page_rank_multiplier = 20

    def __init__(self, inverted_index, n_pages, page_ranks, docs_length={}, inverted_already_tf_idf=False,
                 use_cosine_sim=True):
        self.inverted_index = inverted_index
        self.n_pages = n_pages
        self.page_ranks = page_ranks
        self.idf = self.compute_idf()
        self.use_cosine_sim = use_cosine_sim
        self.doc_length = docs_length
        if not inverted_already_tf_idf:
            self.compute_all_tf_idf()

    def tf_idf(self, word, doc):
        self.inverted_index[word][doc] = self.inverted_index[word][doc] * self.idf[word]
        return self.inverted_index[word][doc]

    def compute_idf(self):
        df = {}
        idf = {}
        for key in self.inverted_index.keys():
            df[key] = len(self.inverted_index[key].keys())
            idf[key] = math.log(self.n_pages / df[key], 2)
        return idf

    def inner_product_similarities(self, query):
        similarity = {}
        for word in query:
            wq = self.idf.get(word, 0)
            if wq != 0:
                for doc in self.inverted_index[word].keys():
                    similarity[doc] = similarity.get(doc, 0) + self.inverted_index[word][doc] * wq
        return similarity

    def inner_product_similarities_expanded(self, query_tokens, query_expansion_tokens):
        similarity = {}
        for word in query_tokens:
            wq = self.idf.get(word, 0)
            if wq != 0:
                for doc in self.inverted_index[word].keys():
                    similarity[doc] = similarity.get(doc, 0) + self.inverted_index[word][doc] * wq
        for word in query_expansion_tokens:
            wq = self.idf.get(word, 0)
            if wq != 0:
                for doc in self.inverted_index[word].keys():
                    similarity[doc] = similarity.get(doc, 0) + self.inverted_index[word][doc] * E_CONST * wq
        return similarity

    def compute_lengths(self, docs_tokens):
        for code in range(self.n_pages):
            self.doc_length[code] = self.compute_doc_length(code, docs_tokens[code])
        return self.doc_length

    def compute_doc_length(self, code, tokens):
        words_accounted_for = []
        length = 0
        for token in tokens:
            if token not in words_accounted_for:
                length += self.tf_idf(token, code) ** 2
                words_accounted_for.append(token)
        return math.sqrt(length)

    def query_length(self, query):
        length = 0
        cnt = Counter()
        for w in query:
            cnt[w] += 1
        for w in cnt.keys():
            length += (cnt[w]*self.idf.get(w, 0)) ** 2
        return math.sqrt(length)

    def cosine_similarities(self, query):
        similarity = self.inner_product_similarities(query)
        for doc in similarity.keys():
            similarity[doc] = similarity[doc] / self.doc_length[doc] / self.query_length(query)
        return similarity

    def cosine_similarities_expanded(self, query_tokens, query_expansion_tokens):
        similarity = self.inner_product_similarities_expanded(query_tokens, query_expansion_tokens)
        for doc in similarity.keys():
            similarity[doc] = similarity[doc] / self.doc_length[doc] / self.query_length(query_tokens)
        return similarity

    def cosine_page_rank(self, query_tokens):
        cosine_similarity = self.cosine_similarities(query_tokens)
        cosine_page_rank_sim = {key: cosine_similarity[key]+self.page_ranks[key]*TfidfRanker.page_rank_multiplier
                                for key in cosine_similarity}
        return cosine_page_rank_sim

    def retrieve_most_relevant(self, query_tokens, use_page_rank_early=False):
        if use_page_rank_early:
            return rank_docs(self.cosine_page_rank(query_tokens))
        else:
            return rank_docs(self.cosine_similarities(query_tokens))

    def retrieve_most_relevant_expanded(self, query_tokens, query_expansion_tokens):
        return rank_docs(self.cosine_similarities_expanded(query_tokens, query_expansion_tokens))

    def compute_all_tf_idf(self):
        for word in self.inverted_index:
            for doc_key in self.inverted_index[word]:
                self.inverted_index[word][doc_key] = self.tf_idf(word, doc_key)



