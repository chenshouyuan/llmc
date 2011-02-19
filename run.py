import os, os.path
from llmc.model.runners import LDARunner, HDPRunner, DPSMMRunner
from llmc.model.util import Corpus, ContinousTimeGaussianGenerator

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

def run_dpsmm(args):
  from llmc.model.util import plot_mode, vi_distance
  gen = ContinousTimeGaussianGenerator(time_len=1,
        sigma1=args['sigma_prior'], sigma2=args['sigma'])
  gen.generate()
  runner = DPSMMRunner(points=gen.data[0], **args)
  out_assign = runner.run() # is a dict
  out_assign = [out_assign[k] for k in xrange(len(out_assign))]
  print "variation information %lf" % vi_distance(gen.assign[0], out_assign)
  plot_mode(gen.assign[0], out_assign, gen.data[0])

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

  p = sub.add_parser('dpm')
  p.set_defaults(function=run_dpsmm)
  p.add_argument('--alpha', action='store', type=float, default=1.0)
  p.add_argument('--sigma_prior', action='store', type=float, default=15.0)
  p.add_argument('--sigma', action='store', type=float, default=1.0)

  p = sub.add_parser('profile')
  p.set_defaults(function=run_profile)

  parser.add_argument('--total_iteration', '--iter', dest='total_iteration',
                      action='store', type=int, default=1000)

  args = parser.parse_args()
  arg_dict = args.__dict__.copy()
  arg_dict.pop('function')
  args.function(arg_dict)
