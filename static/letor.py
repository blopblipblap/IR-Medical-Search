from cmath import cos
import random
import os
import sys
import lightgbm
import numpy as np
import pickle

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

NUM_NEGATIVES = 1
NUM_LATENT_TOPICS = 200

class LETOR:
    def __init__(self, model_dir):
        sys.path.append(r'static')
        self.model_dir = model_dir

    def vector_rep(self, text):
        if hasattr(self, 'model') == False:
            with open(os.path.join(self.model_dir, 'lsi_model.sav'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.model_dir, 'bow_corpus.dict'), 'rb') as f:
                self.dictionary = pickle.load(f)
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]

    def predict(self, query, docs):
        with open(os.path.join(self.model_dir, 'lightgbm_model.sav'), 'rb') as f:
            self.ranker = pickle.load(f)

        X_unseen = []

        for doc_id, doc, doc_real in docs:
            X_unseen.append(self.features(query.split(), doc.split()))
        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _, _) in docs], scores, [doc for (_, _, doc) in docs])]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        return sorted_did_scores

if __name__ == '__main__':
    LETOR_instance = LETOR(model_dir='model')
    LETOR_instance.training()


    




