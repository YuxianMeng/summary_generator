import json
import numpy as np
from scipy.spatial import distance
from gensim.models.keyedvectors import Word2VecKeyedVectors


class feature_extract:
    def __init__(self, threshold=0.5, word2vecpath="model/word_embedding/embedding.wv",
                 datapath="news_ch_2_seg/7.json"):
        self.threshold = threshold
        self.stopword2tag = {'m', 'p', 'x', 'c', 'uj', 'd', 'f', 'r', 'ul'}
        self.stopword2tag.add('a')
        self.word2vec = Word2VecKeyedVectors.load(word2vecpath)
        with open(datapath, 'r') as load_f:
            self.data = json.load(load_f)
        self.content, self.title, self.label = [], [], []
        self.Xtrain = None
        self.init()
        self.de_stopword()
        self.vectorize()

    def init(self):
        self.content = self.data['content'][:]

        self.title = self.data['title'][:]

        self.label = self.data['abstracitve_summary'][:]  # ['extractive_summary_index']

        n = len(self.content)

        self.Xtrain = np.zeros((n, 5))
        self.Xtrain[:, 1] = np.array([i + 1 for i in range(n)])
        self.Xtrain[:, 2] = np.array([len(self.content[i]) for i in range(n)])

    def de_stopword(self):
        for txt in [self.content, self.label]:
            for index, i in enumerate(txt):
                tmp = []
                for j in i:
                    if j[1] not in self.stopword2tag:
                        tmp.append(j)
                txt[index] = tmp

        tmp = []
        for j in self.title:
            if j[1] not in self.stopword2tag:
                tmp.append(j)
        self.title = tmp

    def vectorize(self):
        content2vec = []
        for i in self.content:
            content2vec.append([])
            for j in i:
                content2vec[-1].append(self.get_word_embedding(self.word2vec, j[0]))
        self.content = content2vec

        title2vec = []
        for i in self.title:
            title2vec.append(self.get_word_embedding(self.word2vec, i[0]))
        self.title = title2vec

        label2vec = []
        for i in self.label:
            label2vec.append([])
            for j in i:
                label2vec[-1].append(self.get_word_embedding(self.word2vec, j[0]))
        self.label = label2vec

    def get_word_embedding(self, embed, word):
        if word in embed.vocab:
            return embed.get_vector(word)
        return embed.get_vector('')  # None

    def word_cos_sim(self, w1, w2):
        return 1 - distance.cosine(w1, w2)

    def sentence_unique_sim(self, s1, s2):
        count = 0
        for targetw in s2:
            for candidatew in s1:
                if abs(self.word_cos_sim(candidatew, targetw)) > self.threshold:
                    count += 1;
                    break
        return count

    def sentence_union_sim(self, s1):
        s1np = np.array(s1)
        s1norm = np.linalg.norm(s1np, axis=1)
        sim_matrix = np.dot(s1np, s1np.T) / s1norm ** 2
        visit = set()
        count = 0
        for i in range(len(s1)):
            if i not in visit:
                for j in range(i + 1, len(s1)):
                    if sim_matrix[i, j] > self.threshold:
                        visit.add(j)
                count += 1
        return count


def print2sen(s):
    tmp = []
    for i in s:
        tmp += i[0]
    print(''.join(tmp))


if __name__ == '__main__':
    FE = feature_extract()
    FE.Xtrain[:, 3] = np.array([FE.sentence_unique_sim(i, FE.title) for i in FE.content])
    FE.Xtrain[:, 4] = np.array([FE.sentence_union_sim(i) for i in FE.content])
    print("Matrix")
    print(FE.Xtrain)
    s1 = FE.data['content'][np.argmax(FE.Xtrain[:, 3])]
    print("Title:")
    print2sen(FE.data['title'])
    print("S1:")
    print2sen(s1)