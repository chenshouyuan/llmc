import sys
class BaseSampler:
  def __init__(self, model, total = 5, dot_progress = False, callback = None):
    self.model = model
    self.total = total
    self.dot_progress = dot_progress
    self.callback = callback

  def inference(self):
    for i in xrange(self.total):
      self.show_progress(i)
      self.model.gibbs_iteration()
      if not self.callback is None:
        self.callback(self.model)
    print ""

  def show_progress(self, i):
    if self.dot_progress:
      step = self.total / 20
      if i % step == 0:
        sys.stdout.write('.')
    else:
      print "iteration: %d" % i


