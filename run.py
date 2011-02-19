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
        sigma1=args['sigma_prior'], sigma2=args['sigma'],
        initial_topic=args['n_cluster'],
        data_per_time=args['n_data'])
  gen.generate()
  for x in ['n_cluster', 'n_data']:
    args.pop(x)
  runner = DPSMMRunner(points=gen.data[0], **args)
  out_assign = runner.run() # is a dict
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

  def _sub_parser(sub, name, function, help = ""):
    p = sub.add_parser(name, help=help)
    p.set_defaults(function=function)
    # adding common arguments
    p.add_argument('--total-iteration', '--iter',
                    help = 'number of iteration used by Gibbs sampler',
                    type=int, default=1000)
    return p


  parser = argparse.ArgumentParser(prog='LLMC Builtin Models')
  sub = parser.add_subparsers()

  p = _sub_parser(sub, 'lda', run_lda, help="Run Latent Dirichlet Allocation")
  p.add_argument('--topic-count', type=int, default=20)
  p.add_argument('--alpha', type=float, default=50.0/5)
  p.add_argument('--beta', type=float, default=0.05)

  p = _sub_parser(sub, 'hdp', run_hdp, help="Run Hierarchical Dirichlet Process")
  p.add_argument('--alpha-table', type=float, default=1.0)
  p.add_argument('--alpha-topic', type=float, default=1.0)
  p.add_argument('--beta', type=float, default=1.0)

  p = _sub_parser(sub, 'dpm', run_dpsmm,
                  help="Run Dirichlet Process Mixture (Spherical Gaussian Mixture)")
  p.add_argument('--alpha', type=float, default=1.0)
  p.add_argument('--sigma-prior', type=float, default=15.0)
  p.add_argument('--sigma', type=float, default=1.0)
  p.add_argument('--n-cluster', type=int, default=2)
  p.add_argument('--n-data', type=int, default=100)

  p = _sub_parser(sub, 'profile', run_profile, help="Profiling against HDP model")

  args = parser.parse_args()
  arg_dict = args.__dict__.copy()
  arg_dict.pop('function')
  args.function(arg_dict)
