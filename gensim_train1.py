import gensim
import os
import collections
import smart_open
import random

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
def read_corpus():
    for i, doc in enumerate(documents):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [i])

train_corpus = list(read_corpus())
model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
a = model.infer_vector("human machine")
b = model.infer_vector(['human', 'machine'])
print(a)
print(b)
print(model.predict_output_word(["human","interface"]))

