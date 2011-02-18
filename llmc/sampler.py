import sys
class BaseSampler:
  def __init__(self, model, total = 5, progress='iter', callback = None):
    self.model = model
    self.total, self.progress, self.callback = total, progress, callback

  def inference(self):
    for i in xrange(self.total):
      self.show_progress(i)
      self.model.gibbs_iteration()
      if not self.callback is None:
        self.callback(self.model)
    print ""

  def show_progress(self, i):
    if self.progress=='dot':
      step = self.total / 20
      if i % step == 0:
        sys.stdout.write('.')
    else:
      print "iteration: %d" % i


