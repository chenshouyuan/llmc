from llmc.model.mixture import TestDPSMM

from nose.tools import *

def test_smm_runner_synset():
  from llmc.model.util import ContinousTimeGaussianGenerator as GG
  from llmc.model.util import vi_distance
  from llmc.model.runners import DPSMMRunner
  gen = GG(initial_topic=5, sigma1=30.0, sigma2=1.0, time_len=1,
           data_per_time=100)
  gen.generate()
  runner = DPSMMRunner(points=gen.data[0],sigma_prior=30, sigma=1.0, total_iteration=100)
  out_assign = runner.run()
  variation_information = vi_distance(gen.assign[0], out_assign)
  assert variation_information <= 0.3
