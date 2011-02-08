from sparse import FixedTopicModel
from sampler import BaseSampler
import operator

def topic_statistic_export(topic_freq, vocab, output_path, max_topic = 10):
  with open(output_path, 'w') as f:
    for i, topic in enumerate(topic_freq):
      words = sorted(topic.items(), key=operator.itemgetter(1), reverse=True)[0:max_topic]
      f.write("id: %d\n" % i)
      f.write("\n".join(["%s %d" % (vocab[w], count) for w, count in words]))
      f.write("\n")

class LDARunner:
  def __init__(self, output_path, docs,
      vocab, k = 20, alpha = 0.1, beta = 0.1):
    self.output_path = output_path
    self.docs, self.vocab = docs, vocab
    self.k, self.alpha, self.beta = k, alpha, beta
    self.model = FixedTopicModel(k, len(vocab), alpha, beta)
    for doc in docs:
      self.model.add_new_document(doc)
    self.sampler = BaseSampler(self.model)

  def run(self):
    self.sampler.inference()
    self.save_result()

  def save_result(self):
    exported = self.model.export_assignment()
    topic_freq = [dict() for x in xrange(self.k)]
    for assign, word_list in zip(exported, self.docs):
      for topic, word in zip(assign, word_list):
        topic_freq[topic][word] = topic_freq[topic].get(word,0) + 1
    topic_statistic_export(topic_freq, self.vocab, self.output_path)
