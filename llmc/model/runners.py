from topicmodel import FixedTopicModel, HDPTopicModel
from llmc.sampler import BaseSampler
import operator

def topic_statistic_export(topic_freq, vocab, output_path, max_topic = 10):
  with open(output_path, 'w') as f:
    f.write("%d topics\n" % len(topic_freq))
    for i, topic in enumerate(topic_freq):
      words = sorted(topic.items(), key=operator.itemgetter(1), reverse=True)[0:max_topic]
      f.write("id: %d\n" % i)
      f.write("\n".join(["%s %d" % (vocab[w], count) for w, count in words]))
      f.write("\n")

def save_topic_matrix(model, vocab_list, output_path):
  topic_mat = model.export_topic()
  topic_freq = [dict() for x in xrange(topic_mat.shape[0])]
  for row in xrange(topic_mat.shape[0]):
    dok = topic_mat.getrow(row).todok()
    for _p, _v in dok.iteritems():
      _w = _p[1]
      topic_freq[row][_w] = _v
  topic_statistic_export(topic_freq, vocab_list, output_path)

class LDARunner:
  def __init__(self, output_path, docs,
      vocab, topic_count = 20, alpha = 0.1, beta = 0.1,
      total_iteration=1000):
    self.output_path = output_path
    self.vocab = vocab
    self.k, self.alpha, self.beta = topic_count, alpha, beta
    self.model = FixedTopicModel(topic_count, len(vocab), alpha, beta)
    for doc in docs:
      self.model.add_new_document(doc)
    self.sampler = BaseSampler(self.model, total_iteration)

  def run(self):
    self.sampler.inference()
    save_topic_matrix(self.model, self.vocab, self.output_path)

def _hdp_show_statistics(model):
  print "topics: %d, tables: %d" % (model.topic_count(), model.table_count())

class HDPRunner(LDARunner):
  def __init__(self, output_path, docs, vocab,
               alpha_table=1.0, alpha_topic=1.0, beta=0.5,
               initial_topic=1, initial_table=1,
               total_iteration=1000):
    self.output_path, self.vocab = output_path, vocab
    self.model = HDPTopicModel(len(vocab),
                   alpha_table, alpha_topic, beta,
                   initial_topic, initial_table)
    for doc in docs:
      self.model.add_new_document(doc)
    self.sampler = BaseSampler(self.model,total_iteration,callback=_hdp_show_statistics)

from mixture import DPSMM

def _dpm_show(model):
  print model.cluster_count()

class DPSMMRunner:
  def __init__(self, points, dim=2, alpha=1.0,
               sigma=1.0, sigma_prior=15.0,
               total_iteration=5000):
    self.model = DPSMM(dim=dim, alpha=alpha, sigma=sigma, sigma_prior=sigma_prior)
    for p in points:
      self.model.add_ob(p)
    print self.model.cluster_count()
    self.sampler = BaseSampler(self.model, total=total_iteration,\
                               progress='dot')

  def run(self):
    self.sampler.inference()
    out_assign = self.model.export_assignment()
    return [out_assign[k] for k in xrange(len(out_assign))]
