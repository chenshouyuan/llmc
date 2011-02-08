import sys
class BaseSampler:
  def __init__(self, model, total = 1000, dot_progress = False):
    self.model = model
    self.total = total
    self.dot_progress = dot_progress

  def inference(self):
    for i in xrange(self.total):
      self.show_progress(i)
      self.model.gibbs_iteration()
    print ""

  def show_progress(self, i):
    if self.dot_progress:
      step = self.total / 20
      if i % step == 0:
        sys.stdout.write('.')
    else:
      print "iteration: %d" % i


