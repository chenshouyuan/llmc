import os, os.path
from llmc.topic_model import LDARunner, HDPRunner

# utilty for parsing input data
class Corpus:
  def __init__(self, vocab, train_corpus, test_corpus, stopwords, min_freq = 50, max_freq = 5000):
    self.stopwords, self.min_freq, self.max_freq = stopwords, min_freq, max_freq
    self.read_vocab(vocab)
    self.train_docs, self.train_word_count = self.read_corpus(train_corpus)
    self.test_docs, self.test_word_count = self.read_corpus(test_corpus)
    self.clean_vocab()

  def word_count(self):
    return len(self.vocab)

  def clean_vocab(self):
    with open(self.stopwords,'r') as f:
      stopwords = set([x.strip() for x in f.readlines()])
    self.freq = [0]*len(self.vocab)
    for doc in self.train_docs + self.test_docs:
      self.count_freq(doc)
    clean_index = [i for i,w in enumerate(self.vocab) \
                   if (not self.freq[i] in xrange(self.min_freq, self.max_freq)) \
                       or w in stopwords]
    vocab_map = dict()
    j, k = 0,0
    for i in xrange(len(self.vocab)):
      if j != len(clean_index) and i == clean_index[j]:
        vocab_map[i] = None
        j += 1
      else:
        vocab_map[i] = k
        k += 1

    self.train_docs = [[vocab_map[x] for x in doc if vocab_map[x] != None] \
                        for doc in self.train_docs]
    self.test_docs = [[vocab_map[x] for x in doc if vocab_map[x] != None] \
                        for doc in self.test_docs]
    self.train_word_count = sum([len(x) for x in self.train_docs])
    self.test_word_count = sum([len(x) for x in self.test_docs])
    self.vocab = [self.vocab[i] for i in xrange(len(self.vocab)) if vocab_map[i] != None]
  def count_freq(self, doc):
    for word in doc:
      self.freq[word] += 1

  def read_vocab(self, vocab):
    with open(vocab, 'r') as f:
      self.vocab = [x.rstrip() for x in f.readlines()]

  def read_corpus(self, corpus):
    with open(corpus, 'r') as f:
      doc_count = int(f.readline().rstrip())
      word_count = int(f.readline().rstrip())
      docs = [None] * doc_count
      for i in xrange(doc_count):
        count = int(f.readline().rstrip())
        word_list = [int(x) for x in f.readline().split(' ')]
        docs[i] = word_list
      return (docs, word_count)


def run_topic_model(mode="LDA"):
  base = os.path.split(__file__)[0]
  abspath = os.path.abspath(base)
  datapath = os.path.join(abspath, 'data')
  try:
    os.mkdir(datapath)
  except:
    pass
  traindata = os.path.join(datapath, 'train.kos.txt')
  testdata = os.path.join(datapath, 'test.kos.txt')
  vocabdata = os.path.join(datapath, 'vocab.kos.txt')
  outputdata = os.path.join(datapath, 'topic.kos.txt')
  stopwords = os.path.join(datapath, 'stopwords.txt')
  corpus = Corpus(vocabdata, traindata, testdata, stopwords)
  if mode == "LDA":
    runner=LDARunner(outputdata, corpus.train_docs, corpus.vocab,
                     k = 50, alpha = 50.0 / 20, beta = 0.1)
  else:
    runner=HDPRunner(outputdata, corpus.train_docs, corpus.vocab,
                     total_iteration = 10)
  runner.run()

if __name__ == '__main__':
  #run_topic_model(mode="LDA")
  run_topic_model(mode="HDP")

