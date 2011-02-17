import os, os.path
from llmc.builtin.runners import LDARunner, HDPRunner

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

def _get_corpus(tag = 'kos'):
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
  return outputdata, corpus

def run_lda(args):
  outputdata, corpus = _get_corpus()
  runner=LDARunner(outputdata, corpus.train_docs, corpus.vocab, **args)
  runner.run()

def run_hdp(args):
  outputdata, corpus = _get_corpus()
  runner=HDPRunner(outputdata, corpus.train_docs, corpus.vocab, **args)
  runner.run()

def run_profile(args):
  import pstats, cProfile
  outputdata, corpus = _get_corpus()
  runner=HDPRunner(outputdata, corpus.train_docs, corpus.vocab, total_iteration=3)
  cProfile.runctx('runner.run()', globals(), locals(), "Profile.prof")
  s = pstats.Stats("Profile.prof")
  s.strip_dirs().sort_stats("time").print_stats()

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(prog='LLMC Builtin Models')
  sub = parser.add_subparsers()

  p = sub.add_parser('lda')
  p.set_defaults(function=run_lda)
  p.add_argument('--topic_count', action='store', type=int, default=20)
  p.add_argument('--alpha', action='store', type=float, default=50.0/5)
  p.add_argument('--beta', action='store', type=float, default=0.05)

  p = sub.add_parser('hdp')
  p.set_defaults(function=run_hdp)
  p.add_argument('--alpha_table', action='store', type=float, default=1.0)
  p.add_argument('--alpha_topic', action='store', type=float, default=1.0)
  p.add_argument('--beta', action='store', type=float, default=5.0)

  p = sub.add_parser('profile')
  p.set_defaults(function=run_profile)

  parser.add_argument('--total_iteration', '--iter', dest='total_iteration',
                      action='store', type=int, default=1000)

  args = parser.parse_args()
  arg_dict = args.__dict__.copy()
  arg_dict.pop('function')
  args.function(arg_dict)
