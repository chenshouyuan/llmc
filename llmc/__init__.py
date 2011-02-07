"""
LLMC: A Low Level Markov Chain Monte Carlo Framework

A simple toolkit for implementing Gibbs sampling algorithm
for (discrete) graphic model, e.g. LDA, DPM, HDP. Unlike
previous frameworks, this toolkit would _not_ automatically
calculate the form of posterior distribution. Users should
explictly provide the sampling distribution. In meanwhile,
LLMC provides:
  1) efficient maintainence of sufficient statistics
  2) serializing/deserializing model variables
  3) automatic control of burning, sampling etc.
  4) a unit testing environment
"""


