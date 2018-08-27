# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: summarizer.py
@time: 2018/8/27 11:17

    LexRank Summarizer
    Ref:
    LexRank: Graph-based Lexical Centrality as Salience in Text Summarization.
"""

import math
from collections import Counter, defaultdict

import numpy as np

from algorithms.power_method import stationary_distribution
from utils.text import tokenize


class LexRank:
    def __init__(self, documents, stopwords=None, keep_numbers=False, keep_emails=False,
                   keep_urls=False, include_new_words=True, ):
        """
        :param documents: list:[str]
        :param stopwords:
        :param include_new_words:
        """
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words
        self.idf_score = self._calculate_idf(documents)

    def get_summary_and_keywords(self, sentences, summary_size=1, threshold=.03, fast_power_method=True,
                      return_summary=True, return_keywords=False):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')
        sentences_new = [self.tokenize_sentence(sentence) for sentence in sentences]  # 去停用词和分词

        summary, keywords=None, None
        # 计算关键句
        if return_summary:
            lex_scores = self.rank_sentences(
                sentences_new,
                threshold=threshold,
                fast_power_method=fast_power_method,
            )

            sorted_ix = np.argsort(lex_scores)[::-1]
            summary = [sentences[i] for i in sorted_ix[:summary_size]]
            print(sorted_ix[:summary_size])
            print(np.sort(lex_scores)[::-1])

        # 计算关键词
        if return_keywords:
            pass

        return summary, keywords

    def get_summary(self, sentences, summary_size=1, threshold=.03, fast_power_method=True,):
        summary, _ = self.get_summary_and_keywords(sentences, summary_size=summary_size, threshold=threshold,
                                                   fast_power_method=fast_power_method,
                                                   return_summary=True, return_keywords=False)
        return summary

    def get_keywords(self, sentences, summary_size=1, threshold=.03, fast_power_method=True,):
        _, keywords = self.get_summary_and_keywords(sentences, summary_size=summary_size, threshold=threshold,
                                                    fast_power_method=fast_power_method,
                                                    return_summary=False, return_keywords=True)
        return keywords

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )


        tf_scores = [Counter(s) for s in sentences]

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores


    def tokenize_sentence(self, sentence):
        tokens = tokenize(
            sentence,
            self.stopwords,
            keep_numbers=self.keep_numbers,
            keep_emails=self.keep_emails,
            keep_urls=self.keep_urls,
        )

        return tokens

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = math.log(doc_number_total + 1)

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def _calculate_similarity_matrix(self, tf_scores):
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j)
                # TODO: 添加更多的相似度，如embedding和keyword-based

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

    def _markov_matrix(self, similarity_matrix):
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        return similarity_matrix / row_sum

    def _markov_matrix_discrete(self, similarity_matrix, threshold):
        markov_matrix = np.zeros(similarity_matrix.shape)

        for i in range(len(similarity_matrix)):
            columns = np.where(similarity_matrix[i] > threshold)[0]
            markov_matrix[i, columns] = 1 / len(columns)

        return markov_matrix


if __name__ == '__main__':
    from path import Path
    documents = []
    documents_dir = Path('/data/shannon/yuxian/data/lexrank_test/bbc/politics')

    for file_path in documents_dir.files('*.txt'):
        with file_path.open(mode='rt', encoding='utf-8') as fp:
            documents.append(fp.readlines())

    lxr = LexRank(documents)

    sentences = [
        'One of David Cameron\'s closest friends and Conservative allies, '
        'George Osborne rose rapidly after becoming MP for Tatton in 2001.',

        'Michael Howard promoted him from shadow chief secretary to the '
        'Treasury to shadow chancellor in May 2005, at the age of 34.',

        'Mr Osborne took a key role in the election campaign and has been at '
        'the forefront of the debate on how to deal with the recession and '
        'the UK\'s spending deficit.',

        'Even before Mr Cameron became leader the two were being likened to '
        'Labour\'s Blair/Brown duo. The two have emulated them by becoming '
        'prime minister and chancellor, but will want to avoid the spats.',

        'Before entering Parliament, he was a special adviser in the '
        'agriculture department when the Tories were in government and later '
        'served as political secretary to William Hague.',

        'The BBC understands that as chancellor, Mr Osborne, along with the '
        'Treasury will retain responsibility for overseeing banks and '
        'financial regulation.',

        'Mr Osborne said the coalition government was planning to change the '
        'tax system \"to make it fairer for people on low and middle '
        'incomes\", and undertake \"long-term structural reform\" of the '
        'banking sector, education and the welfare state.',
    ]

    # get summary with classical LexRank algorithm
    summary = lxr.get_summary(sentences, summary_size=2, threshold=0.03)
    print(summary)
    #[0.99317452 0.97779086 1.00605786 1.01041131 1.00011702 0.99523021 1.01721823]

    # get summary with continuous LexRank
    summary_cont = lxr.get_summary(sentences, threshold=None)  # 当新闻不是很长的时候，目测continous比较稳定
    print(summary_cont)


    # # 'fast_power_method' speeds up the calculation, but requires more RAM
    # scores_cont = lxr.rank_sentences(
    #     sentences,
    #     threshold=.1,
    #     fast_power_method=False,
    # )
    # print(scores_cont)
