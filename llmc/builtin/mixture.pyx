from llmc.spmatrix cimport *
from llmc.spmatrix import *
from llmc.modelutil cimport *

from libc.stdlib cimport free, malloc, rand, RAND_MAX
from libc.math cimport log, exp
from libc.stdio cimport printf

cdef:
  # the most simple mixture model
  # only mean has normal prior
  # x_i ~ N(\mu_c_i, \sigma*I)
  # \mu_i ~ N(0, \sigma_prior*I)

  DEF MAX_DIM = 3

  struct simple_mixture_model:
    matrix *mp, *mc  # points, clusters
    matrix_mult_view *view
    int dim
    double sigma_prior, sigma 
    sample_buffer buf[200000]

  simple_mixture_model* ssm_new(double sigma_prior,double sigma):
    cdef simple_mixture_model *ssm = <simple_mixture_model*>malloc(sizeof(simple_mixture_model))
    ssm.mp = matrix_new(0, 0)
    ssm.mc = matrix_new(1, 0)
    cdef matrix *prod = matrix_new(0, 0)
    ssm.view = mult_view_new(ssm.mc, ssm.mp, prod)
    ssm.sigma_prior = sigma_prior
    ssm.sigma = sigma
    return ssm
  
  void ssm_free(simple_mixture_model *m):
    matrix_delete(m.mp)
    matrix_delete(m.mc)
    matrix_delete(m.view.prod)
    mult_view_delete(m.view)
    free(m)

  void _extract_cord(int dim, double *point, matrix *m, vector *row):
    cdef int i
    cdef _ll_item *p = m.cols.head.next
    cdef vector *col
    for i in range(dim):
      col = <vector*> p.data
      point[i] = get_matrix_entry(row, col)
      p = p.next

  double _loglike_normal_simple(int dim, double *mu, double var, double *ob):
    cdef double sum = 0
    for i in range(dim):
      sum += (mu[i]-ob[i])*(mu[i]-ob[i])
    return -0.5*log(var)-sum/(2*var)    
  
  double _log_prior_ssm(simple_mixture_model *m, vector *row_mp):
    cdef double ob[MAX_DIM]
    cdef double zeros[MAX_DIM]
    for i in range(m.dim):
      zeros[i] = 0
    _extract_cord(m.dim, ob, m.mp, row_mp)
    cdef double var = m.sigma_prior*m.sigma_prior+1
    return _loglike_normal_simple(m.dim, zeros, var, ob)

  double _log_posterior_ssm(simple_mixture_model *m,\
      entry_t count, vector *row_prod, vector *row_mp):
    cdef double mu[MAX_DIM], ob[MAX_DIM]
    if count == 0:
      return 0    
    _extract_cord(m.dim, mu, m.view.prod, row_prod)
    _extract_cord(m.dim, ob, m.mp, row_mp)
    cdef int i
    for i in range(m.dim):
      mu[i] /= count
    cdef double mu_var = 1.0/(1.0/(m.sigma*m.sigma)+count)
    return _loglike_normal_simple(m.dim, mu, mu_var+1, ob)

  int _get_sample_buffer_dpssm(simple_mixture_model *m, vector *col_mc, double log_alpha):
    cdef _ll_item *p = m.mc.rows.head.next
    cdef vector *row_mc, *row_prod, *row_mp
    cdef entry_t value
    cdef int count = 0
    row_mp = mult_view_map_to_right(m.view, col_mc)
    while p:
      row_mc = <vector*> p.data
      value = get_matrix_entry(row_mc, col_mc)
      row_prod = mult_view_map_prod_row(m.view, row_mc)
      m.buf[count].prob = log(row_mc.sum)+\
        _log_posterior_ssm(m, row_mc.sum, row_prod, row_mp)
      m.buf[count].ptr = row_mc
      count += 1
      p = p.next
    m.buf[count].prob = log_alpha+_log_prior_ssm(m, row_mp)
    m.buf[count].ptr = NULL
    count += 1
    return count

  void resample_dpsmm(simple_mixture_model *m, double log_alpha):
    cdef _ll_item *p = m.mc.cols.head.next
    cdef vector *col_mc, *row_mc
    cdef int count, i
    while p:
      col_mc = <vector*> p.data
      row_mc = _get_first(col_mc).row
      matrix_update(m.mc, -1, row_mc, col_mc)
      count = _get_sample_buffer_dpssm(m, col_mc, log_alpha)
      row_mc = <vector*>sample_log_unnormalized(m.buf, count)
      if row_mc is NULL:
        row_mc = matrix_insert_new_row(m.mc)
      matrix_update(m.mc, +1, row_mc, col_mc)
      p = p.next    

import random

cdef class DPSMM:
  cdef simple_mixture_model *m
  cdef object mp_cols, mc_rows, mp_rows
  cdef double log_alpha

  def __init__(self, dim=2, initial_clusters=1, sigma_prior=15.0, sigma=1.0, alpha=1.0):
    self.m = ssm_new(sigma_prior, sigma)
    self.log_alpha = log(alpha)
    self.set_dim(dim)
    self.set_cluster(initial_clusters)
    self.mp_rows = []

  def set_cluster(self, cluster_count):
    self.mc_rows = [<int>matrix_insert_new_row(self.m.mc) for x in xrange(cluster_count)]

  def set_dim(self, dim):
    self.m.dim = dim
    self.mp_cols = [<int>matrix_insert_new_col(self.m.mp) for x in xrange(dim)]

  def add_ob(self, ob, debug_row=None):
    assert <int>len(ob) == <int>self.m.dim
    cdef vector *mc_col = matrix_insert_new_col(self.m.mc)
    cdef vector *mp_row = mult_view_map_to_right(self.m.view, mc_col)
    self.mp_rows.append(<int>mp_row)
    cdef vector *mp_col
    for _col, value in zip(self.mp_cols, ob):
      mp_col = <vector*><int>_col
      matrix_update(self.m.mp, value, mp_row, mp_col)
    cdef vector *mc_row   
    if debug_row is None:
      _row = random.choice(self.mc_rows)
    else:
      _row = self.mc_rows[debug_row]
    mc_row = <vector*><int> _row
    matrix_update(self.m.mc, 1.0, mc_row, mc_col)

  def gibbs_iteration(self):
    resample_dpsmm(self.m, self.log_alpha)

  def export_assignment(self):
    mat = to_scipy_matrix(<int>self.m.mc)
    assign = dict()
    for row,col in mat.iterkeys():
      assign[col]=row
    return assign
  
  def cluster_count(self):
    return self.m.mc.row_count

################
# unit testing #
################
import unittest
from nose.tools import *
import numpy as np
from numpy.random import multivariate_normal

def _sq(x):
  return np.dot(x,x)

def _log_gaussian(mean, var, x):
  return -0.5*log(var)-_sq(x-mean)/(2*var)

class TestDPSSM:
#  def tearDown(self):
#    ssm_free(self.model.m)
  
  def _test_extract_cord(self, DPSMM model):   
    cdef simple_mixture_model *m = model.m    
    cdef double buf[2]
    eq_(len(self.points), len(model.mp_rows))
    for i, p in enumerate(self.points):
      _extract_cord(self.dim, buf, m.mp, <vector*><int>model.mp_rows[i])
      for k in xrange(self.dim):
        eq_(p[k], buf[k])

  def _test_cluster_assign(self, DPSMM model):
    mat = to_scipy_matrix(<int>model.m.mc)
    for row,col in mat.iterkeys():
      eq_(row, self.assign[col])

  def test_sampler_buffer(self):    
    cdef DPSMM model    
    self.dim, self.cluster_count = 2, 3
    model = DPSMM(dim=self.dim, initial_clusters=self.cluster_count)    
    #self.points = [[5,5], [2,3], [4,5], [10, 20], [40,50]]
    self.points = [multivariate_normal(np.zeros(self.dim), np.identity(self.dim))\
                   for x in xrange(10)]
    self.assign = [random.randint(0, self.cluster_count-1) for p in self.points]
    for p,c in zip(self.points, self.assign):
      model.add_ob(p, debug_row = c)
 
    self._test_cluster_assign(model)
    self._test_extract_cord(model)

    cdef simple_mixture_model *m = model.m
    mc_mat = to_scipy_matrix(<int>m.mc)
    mp_mat = to_scipy_matrix(<int>m.mp)

    mc_cols = to_data_array(<int>m.mc.cols)
    m.mc.squeeze_row = 0

    for i in xrange(len(self.points)):
      _col,_row = mc_cols[i], model.mc_rows[self.assign[i]]
      col = <vector*><int>_col
      row = <vector*><int>_row
      matrix_update(m.mc, -1, row, col)
      eq_(mc_mat[self.assign[i], i], 1)
      mc_mat[self.assign[i], i] -= 1
      prod = np.dot(mc_mat, mp_mat).toarray()
      _get_sample_buffer_dpssm(m, col, model.log_alpha)
      for k in xrange(self.cluster_count):
        count = mc_mat[k, :].sum()
        if count == 0:
          continue
        truth = log(count) \
          +_log_gaussian(mean=prod[k,:]/count,\
                         var=1.0/(1.0/(m.sigma*m.sigma)+count)+1,
                         x=self.points[i])
        assert_almost_equal(truth, m.buf[k].prob)
        eq_(<int>m.buf[k].ptr, model.mc_rows[k])
      truth = model.log_alpha+\
          _log_gaussian(np.zeros((self.dim)), m.sigma_prior*m.sigma_prior+1,\
                        self.points[i])
      assert_almost_equal(m.buf[self.cluster_count].prob, truth)
      eq_(<int>m.buf[self.cluster_count].ptr, 0)
      matrix_update(m.mc, +1, row, col)
      mc_mat[self.assign[i], i] += 1  
