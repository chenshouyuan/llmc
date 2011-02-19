"""
utility modules used by builtin models
  for parsing input, evaluating output etc.
"""



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


def _setattrs(obj, *dicts):
  for dic in dicts:
    for k in dic:
      setattr(obj,k,dic[k])

import numpy as np
import numpy.random

def _gaussian_sample(mean, sigma):
  cov=sigma*sigma*np.identity(len(mean))
  return np.random.multivariate_normal(mean,cov)

class ContinousTimeGaussianGenerator:
  default_args = \
    { 'living': 0.8,         'sigma1': 10,
      'sigma2': 1,           'time_len': 30,
      'initial_topic': 2,    'dim': 2,
      'data_per_time': 50,   'delta': 1.0,
      'model_delta': 0.1}

  def __init__(self, **kargs):
    _setattrs(self, self.__class__.default_args, kargs)

  def generate(self):
    self.gen_comps()
    self.gen_data()
    self.gen_times()

  def gen_single_comp(self):
    return _gaussian_sample(np.zeros(self.dim),self.sigma1)

  def new_comp_id(self):
    ret = self.total_comp_id
    self.total_comp_id += 1
    return ret

  def gen_comps(self):
    self.total_comp_id = 0
    self.comp_ids = [None]*self.time_len
    self.comp_ids[0] = [self.new_comp_id() for x in xrange(self.initial_topic)]
    for t in xrange(1, self.time_len):
      while True:
        self.comp_ids[t] = [x for x in self.comp_ids[t-1] if random.random()<=self.living]
        new_comp = np.random.poisson((1-self.living)*self.initial_topic)
        self.comp_ids[t] += [self.new_comp_id() for x in xrange(new_comp)]
        if len(self.comp_ids[t]) > 0:
          break
    self.comp_values = [self.gen_single_comp() for x in xrange(self.total_comp_id)]
    self.comp_time = [[v] for v in self.comp_values]
    for t in xrange(self.time_len-1):
      for comp_time in self.comp_time:
        last_value = comp_time[-1]
        comp_time.append(last_value+_gaussian_sample(np.zeros(self.dim),self.delta))

  def gen_single_data(self, comp):
    return _gaussian_sample(comp,self.sigma2)

  def gen_data(self):
    self.data = [None]*self.time_len
    self.assign = [None]*self.time_len
    for t,ids in enumerate(self.comp_ids):
      self.assign[t] = sum([[cid]*self.data_per_time for cid in ids], [])
      #print self.assign[t]
      #self.assign[t] = [random.choice(ids) for x in xrange(self.data_per_time)]
      self.data[t] = [self.gen_single_data(self.comp_time[cid][t]) \
                      for cid in self.assign[t]]

  def gen_times(self):
    self.times = list(xrange(self.time_len))

import matplotlib.pyplot as plt
_cname = \
["#DC1433","#00FFFF","#00008B","#008B8B","#B8860B","#A9A9A9","#006400","#BDB76B",
 "#8B008B","#556B2F","#FF8C00","#9932CC","#8B0000","#E9967A","#8FBC8F","#483D8B",
 "#2F4F4F","#00CED1","#9400D3","#FF1493","#000000","#0000FF","#7FFF00","#5F9EA0"]

def plot_mode(in_assign, out_assign, data):
  def _cc(comp_id):
    def _chash(a):
      a = (a+0x7ed55d16) + (a<<12);
      a = (a^0xc761c23c) ^ (a>>19);
      a = (a+0x165667b1) + (a<<5);
      a = (a+0xd3a2646c) ^ (a<<9);
      a = (a+0xfd7046c5) + (a<<3);
      a = (a^0xb55a4f09) ^ (a>>16);
      return a
    color_id = int(_chash(comp_id) % len(_cname))
    return _cname[color_id]

  def plt_points(assign, data, id_map=lambda x:x, sign="^"):
    for a, d in zip(assign, data):
      plt.plot(d[0], d[1], sign, color=_cc(id_map(a)))

  plt.subplot(2, 1, 1)
  plt_points(in_assign, data)
  plt.title("assignment (ground truth)")
  plt.subplot(2, 1, 2)
  plt_points(out_assign, data)
  plt.title("assignment (dpssm)")
  plt.show()

def vi_distance(in_assign, out_assign):
  from math import log

  def entropy(array):
    count = sum(array)
    return -sum([entropy_term(x,count) for x in array])

  def entropy_term(x, count):
    if x == 0:
      return 0
    t = float(x)/count
    return t*log(t)

  def count_cluster(assign):
    count_dict = dict()
    for a in assign:
      count_dict[a] = count_dict.get(a, 0)+1
    return count_dict

  out_count = count_cluster(out_assign)
  in_count = count_cluster(in_assign)
  n = sum(out_count.values())
  vi_dict = dict()
  for x,y in zip(in_assign, out_assign):
    pair=(x,y)
    vi_dict[pair] = vi_dict.get(pair, 0)+1
  vi = 0
  for x in in_count:
    for y in out_count:
      if (x,y) in vi_dict:
        c = float(vi_dict[(x,y)])
        vi += c*(log(c*n/in_count[x]/out_count[y]))
  vi /= n
  h1 = entropy(out_count.values())
  h2 = entropy(in_count.values())
  return h1+h2-2*vi


