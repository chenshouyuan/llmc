# cython: profile=True

from llmc.spmatrix cimport *
from llmc.spmatrix import *

from libc.stdlib cimport free, malloc, rand, RAND_MAX
from libc.math cimport log, exp
from libc.stdio cimport printf

cdef extern from "gsl/gsl_sf_gamma.h":
  double gsl_sf_lngamma(double x)

DEF MAX_SAMPLE_BUFFER = 2000

cdef:
  struct sample_buffer:
    double prob
    void  *ptr

  struct vec_group:
    int    count
    vector **vec
  
  struct document:
    int     count
    int    *words
  
  struct topic_model:
    int vocab_size
    int doc_count
    document* docs
    matrix *m_vocab  # LDA/HDP: N*W
    matrix *m_docs   # LDA: KD*N, HDP: Tables*N
      # LDA: col.aux => vec_group
      # HDP: col.aux => femap
    matrix *m_topic  # LDA: K*KD, HDP: Topic*Tables
    matrix_mult_view *view_doc_word #m_doc*m_vocab
    matrix_mult_view *view_topic_word #m_topic*m_doc*m_vocab
    sample_buffer buf[MAX_SAMPLE_BUFFER]
  
  inline matrix_entry* _get_first(vector* vec):
    return <matrix_entry*>vec.store.list.head.next.data

  # consider D=A*(B*C) 
  inline vector* _to_l(matrix_mult_view *v1, matrix_mult_view *v2, vector* vec):
    cdef vector* temp
    temp = mult_view_map_prod_row(v1, vec) # row in (B*C)
    temp = _get_first(mult_view_map_to_left(v2, temp)).row # row in A
    return mult_view_map_prod_row(v2, temp) # row in D

  inline vector* _to_r(matrix_mult_view *v1, matrix_mult_view *v2, vector* vec):
    cdef vector* temp
    temp = _get_first(mult_view_map_to_right(v1, vec)).col # col in C
    temp = mult_view_map_prod_col(v1, temp) # col in (B*C)
    return mult_view_map_prod_col(v2, temp) # col in D
  
  void* sample_unnormalized(sample_buffer *buf, int count):
    cdef int i
    cdef double sum = 0
    for i in range(count):
      sum += buf[i].prob
    cdef double coin = (rand()+0.0)/RAND_MAX*sum
    sum = 0
    for i in range(count):
      sum += buf[i].prob
      if sum >= coin:
        return buf[i].ptr

  double _log_sum_normalize(sample_buffer *buf, int count):
    cdef double log_max = 100, max_value=-10000000
    cdef double log_shift, exp_sum, log_norm
    cdef int i
    for i in range(count):
      if buf[i].prob > max_value:
        max_value = buf[i].prob
    log_shift = log_max-log(count+1.0)-max_value
    exp_sum = 0
    for i in range(count):
      exp_sum += exp(buf[i].prob+log_shift)
    log_norm = log(exp_sum)-log_shift
    for i in range(count):
      buf[i].prob -= log_norm
    return log_norm

  void* sample_log_unnormalized(sample_buffer *buf, int count):
    cdef double log_norm = _log_sum_normalize(buf, count)
    cdef int i 
    for i in range(count):
      buf[i].prob = exp(buf[i].prob)
    return sample_unnormalized(buf, count)
  
  # for LDA
  int _get_sample_buffer(topic_model*t, vector* col, double alpha, double beta):
    cdef vector *word, *topic_row, *row
    cdef vec_group *group        
    cdef int i, count
    word = _to_r(t.view_doc_word, t.view_topic_word, col)
    group = <vec_group*> col.aux
    for i in range(group.count):
      row = group.vec[i]    
      topic_row = _to_l(t.view_doc_word, t.view_topic_word, row)
      count = get_matrix_entry(topic_row, word)
      t.buf[i].prob = \
          (row.sum+alpha)*(count+beta)/(topic_row.sum+t.vocab_size*beta)
      t.buf[i].ptr = <void*>row
    return group.count

  # gibbs iteration of LDA
  void resample_topic_model(topic_model* t, double alpha, double beta):
    cdef _ll_item *p = t.m_docs.cols.head.next
    cdef vector *row, *col
    cdef int i, count
    while p:    
      col = <vector*> p.data
      row = <vector*> _get_first(col).row
      matrix_update(t.m_docs, -1, row, col)           
      count = _get_sample_buffer(t, col, alpha, beta)
      row = <row_type*> sample_unnormalized(t.buf, count)      
      matrix_update(t.m_docs, +1, row, col)
      p = p.next

  # for table assignment in HDP
  int _get_sample_buffer_table(topic_model *t, vector *col, double alpha_table, double beta):
    cdef vector *word, *topic_row, *row
    cdef femap *group
    cdef _ll_item *p
    cdef int count = 0, value 
    word = _to_r(t.view_doc_word, t.view_topic_word, col)
    group = <femap*> col.aux
    p = group.list.head.next
    while p:
      row = <vector*> p.data     
      topic_row = _to_l(t.view_doc_word, t.view_topic_word, row)
      value = get_matrix_entry(topic_row, word)
      t.buf[count].prob = \
          row.sum*(value+beta)/(topic_row.sum+t.vocab_size*beta)
      t.buf[count].ptr = <void*>row
      count += 1
      p = p.next    
    t.buf[count].prob = alpha_table*1.0/t.vocab_size
    t.buf[count].ptr = NULL
    count += 1
    return count
  
  # for topic assignment in HDP
  int _get_sample_buffer_topic(topic_model *t, vector *table, double log_alpha_topic, double beta):
    cdef vector *word, *topic_row, *row
    cdef _ll_item *p
    cdef int count = 0
    p = t.m_topic.rows.head.next
    while p:
      row = <vector*> p.data
      topic_row = mult_view_map_prod_row(t.view_topic_word, row)
      t.buf[count].prob = log(row.sum)+_log_posterior_table(t, topic_row, table, beta)      
      t.buf[count].ptr = <void*> row
      count += 1
      p = p.next
    t.buf[count].prob = log_alpha_topic+_log_prior_table(t, table, beta)
    t.buf[count].ptr = NULL
    count += 1
    return count

  inline double ln_factorial(int count, int start, double beta):
    return gsl_sf_lngamma(beta+count+start)-gsl_sf_lngamma(beta+start)

  double _log_prior_table(topic_model *t, vector *table, double beta):
    cdef _ll_item *p = table.store.list.head.next
    cdef vector *col_word # column in Tables*Words matrix
    cdef double ret = 0
    cdef matrix_entry* entry
    while p:
      entry = <matrix_entry*> p.data
      col_word = entry.col
      ret += ln_factorial(get_matrix_entry(table, col_word), 0, beta)
      p = p.next
    ret -= ln_factorial(table.sum, 0, t.vocab_size*beta)
    return ret
    
  double _log_posterior_table(topic_model *t, vector *row_topic_word, vector *table, double beta):
    cdef _ll_item *p = table.store.list.head.next
    cdef vector *col # column in Tables*Words matrix
    cdef matrix_entry* entry
    cdef int topic_count, table_count
    cdef double ret = 0
    while p:
      entry = <matrix_entry*> p.data
      col = entry.col
      topic_count = get_matrix_entry(row_topic_word, mult_view_map_prod_col(t.view_topic_word, col))
      table_count = get_matrix_entry(table, col)
      ret += ln_factorial(table_count, topic_count, beta)
      p = p.next
    ret -= ln_factorial(table.sum, row_topic_word.sum, t.vocab_size*beta)
    return ret

  inline femap* _get_group(_ll_item *p):
    cdef vector* col = <vector*> p.data
    return <femap*>col.aux

  # gibbs iteration of HDP
  void resample_hdp(topic_model *t, double alpha_table, double log_alpha_topic, double beta):
    cdef _ll_item *p = t.m_docs.cols.head.next
    while p:
      p = resample_hdp_docs(t, p, alpha_table, log_alpha_topic, beta)
      resample_hdp_tables(t, _get_group(p), log_alpha_topic, beta)
      p = p.next
      
  _ll_item* resample_hdp_docs(topic_model *t, _ll_item *p, 
                              double alpha_table, double log_alpha_topic, double beta):
    cdef femap* group = NULL    
    cdef vector *row, *col
    cdef int count
    cdef _ll_item *prev = <_ll_item*>0
    while p:
      col = <vector*> p.data
      if group is NULL:
        group = <femap*> col.aux      
      elif col.aux != group:
        return prev
      row = <vector*> _get_first(col).row
      matrix_update(t.m_docs, -1, row, col)
      if row.sum == 0:
        femap_remove(group, row)
        matrix_remove_row(t.m_docs, row)
      count = _get_sample_buffer_table(t, col, alpha_table, beta)
      row = <row_type*> sample_unnormalized(t.buf, count)
      if row is NULL: # add new table
        row = matrix_insert_new_row(t.m_docs)
        femap_insert(group, <void*>row, <void*>row)
        resample_hdp_table(t, row, log_alpha_topic, beta)
      matrix_update(t.m_docs, +1, row, col)
      prev = p
      p = p.next
    return prev

  vector* resample_hdp_table(topic_model *t, vector* row_docs,\
                             double log_alpha_topic, double beta):
    cdef int count
    cdef vector *row_doc_word, *col_topic, *row_topic
    row_doc_word = mult_view_map_prod_row(t.view_doc_word, row_docs) #table
    col_topic = mult_view_map_to_left(t.view_topic_word, row_doc_word) #table
    if col_topic.sum != 0:
      row_topic = <vector*> _get_first(col_topic).row #topic     
      matrix_update(t.m_topic, -1, row_topic, col_topic) # m_topic squeeze automatically    
    count = _get_sample_buffer_topic(t, row_doc_word, log_alpha_topic, beta)
    row_topic = <row_type*> sample_log_unnormalized(t.buf, count)
    if row_topic is NULL: # add new topic
      row_topic = matrix_insert_new_row(t.m_topic)
    matrix_update(t.m_topic, +1, row_topic, col_topic)

  void resample_hdp_tables(topic_model *t, femap* group, double log_alpha_topic, double beta):
    cdef _ll_item *p = group.list.head.next
    cdef vector *row_docs
    cdef int count
    while p:
      row_docs = <vector*> p.data #table
      resample_hdp_table(t, row_docs, log_alpha_topic, beta)
      p = p.next


###################
# wrappers
###################

import random

cdef class FixedTopicModel:
  cdef topic_model *t
  cdef object _vocab_list
  cdef object _topic_rows
  cdef int    _topic_count
  cdef double alpha, beta
  cdef object doc_columns, doc_rows

  def __init__(self, k, vocab_size, alpha, beta):    
    self.initialize_model()

    self.alpha = alpha
    self.beta = beta  
    if not k is None:
      self.set_topic_num(k)    
    if not vocab_size is None:
      self.set_vocab_size(vocab_size)

  def initialize_model(self):
    cdef matrix *prod1, *prod2
    self.t = <topic_model*>malloc(sizeof(topic_model))
    self.t.doc_count = 0
    self.t.vocab_size = 0
    self.t.m_docs = matrix_new(0,0)
    self.t.m_vocab = matrix_new(0,0)
    self.t.m_topic = matrix_new(0,0)

    prod1 = matrix_new(0,0)
    self.t.view_doc_word = mult_view_new(self.t.m_docs, self.t.m_vocab, prod1)

    prod2 = matrix_new(0,0)
    self.t.view_topic_word = mult_view_new(self.t.m_topic, prod1, prod2)
    self.doc_columns = []
    self.doc_rows = []

  def set_topic_num(self, k):
    self._topic_count = <int> k
    self._topic_rows = [<int>matrix_insert_new_row(self.t.m_topic) for i in xrange(k)]

  def set_vocab_size(self, vocab_size):
    self.t.vocab_size = vocab_size
    self._vocab_list = [<int>matrix_insert_new_col(self.t.m_vocab) for i in xrange(vocab_size)]

  def add_new_document(self, word_list, debug_m_docs=None, debug_m_topic=None):
    cdef vector *topic_row, *row, *col, *temp
    cdef void *group
    self.t.doc_count += 1
    topics, _g = self._set_m_topic(debug_m_topic)
    group = <void*><int>_g
    cols = self._set_m_vocab(word_list)
    self._set_m_doc(<int>group, cols, topics, debug_m_docs)

  def _set_m_topic(self, debug_m_topic=None):
    cdef vec_group *group = <vec_group*>malloc(sizeof(vec_group))
    topics = [<int>matrix_insert_new_row(self.t.m_docs) for _x in xrange(self._topic_count)]
    group.count = self._topic_count
    group.vec = <vector**>malloc(sizeof(vector*)*group.count)
    cdef int i = 0
    for _topic_row, _row in zip(self._topic_rows, topics):
      row = <vector*> <int> _row
      topic_row = <vector*> <int> _topic_row
      temp = mult_view_map_prod_row(self.t.view_doc_word, row)
      col = mult_view_map_to_left(self.t.view_topic_word, temp)
      matrix_update(self.t.m_topic, +1, topic_row, col)
      group.vec[i] = row
      i += 1
    self.doc_rows.append(topics)    
    return (topics, <int>group)

  def _set_m_vocab(self, word_list):
    word_cols = [<int>matrix_insert_new_col(self.t.m_docs) for w in word_list]    
    for _col, word in zip(word_cols, word_list):
      col = <vector*><int> _col
      row = mult_view_map_to_right(self.t.view_doc_word, col)
      temp = <vector*><int> self._vocab_list[word]
      matrix_update(self.t.m_vocab, +1, row, temp)          
    self.doc_columns.append(word_cols)    
    return word_cols
  
  def _set_m_doc(self, _group, cols, rows, debug_m_doc=None):
    cdef void* group = <void*><int> _group
    for i, _col in enumerate(cols):
      col = <vector*><int> _col
      col.aux = group     
      if debug_m_doc:
        _row = rows[debug_m_doc[i]]
      else:
        _row = random.choice(rows)
      row = <vector*><int> _row
      matrix_update(self.t.m_docs, +1, row, col)    

  def gibbs_iteration(self):
    resample_topic_model(self.t, self.alpha, self.beta)

  def export_assignment(self):
    sparse = to_scipy_matrix(<int>self.t.m_docs)
    topic_map = dict()
    for row, col in sparse.iterkeys():
      topic_map[col] = row
    index = 0
    exported = []
    for i, col in enumerate(self.doc_columns):
      this_doc = []
      for j in xrange(index, index+len(col)):
        topic = topic_map[j] - i*self._topic_count
        this_doc.append(topic)
      index += len(col)
      exported.append(this_doc)
    return exported

  def export_topic(self):
    return to_scipy_matrix(<int>self.t.view_topic_word.prod).tocsr()  

cdef class HDPTopicModel(FixedTopicModel):
  cdef double alpha_table
  cdef double log_alpha_topic
  cdef int initial_topic, initial_table
  cdef object doc_groups

  def __init__(self, vocab_size, alpha_table, alpha_topic, beta,
               initial_topic=1, initial_table=1):
    self.initialize_model()
    self.t.m_topic.squeeze_row = 1
    self.alpha_table = alpha_table
    self.log_alpha_topic = log(alpha_topic)
    self.beta = beta    
    self.initial_topic = initial_topic
    self.initial_table = initial_table
    self.doc_groups = []
    if not vocab_size is None:
      self.set_vocab_size(vocab_size)
    self.set_topic_num(initial_topic)

  def _set_m_topic(self, debug_m_topic=None):
    cdef femap* group = femap_new()
    cdef vector *temp, *col, *table, *topic_row
    tables = [<int>matrix_insert_new_row(self.t.m_docs) for x in xrange(self.initial_table)]
    for i, _table in enumerate(tables):
      table = <vector*><int> _table
      temp = mult_view_map_prod_row(self.t.view_doc_word, table)
      col = mult_view_map_to_left(self.t.view_topic_word, temp)
      if not debug_m_topic is None:
        topic_row = <vector*><int>self._topic_rows[debug_m_topic[i]]
      else:
        topic_row = <vector*><int>random.choice(self._topic_rows)
      matrix_update(self.t.m_topic, +1, topic_row, col)
      femap_insert(group, <void*>table, <void*>table)
    self.doc_rows.append(tables)          
    self.doc_groups.append(<int>group)
    return (tables, <int>group)
 
  def topic_count(self):
    return self.t.m_topic.row_count

  def table_count(self):
    return self.t.m_docs.row_count

  def gibbs_iteration(self):
    resample_hdp(self.t, self.alpha_table, self.log_alpha_topic, self.beta)


#######################################
# unit testing
#######################################

from nose.tools import *
import unittest

import scipy.sparse
import numpy as np
import math

class TestTopicModel:
  def test_topic_model(self):
    vocab_size = 6
    k = 3
    alpha = 0.1
    beta = 0.2
    word_lists = [[0,1,2], [3,4,5], [1,3,5], [0,2,4]]    
    topic_select = [[random.randint(0,k-1) for _x in _y] for _y in word_lists]
    column_indices = []
    _j = 0
    for _y in word_lists:
      column_indices.append(list(range(_j, _j+len(_y))))
      _j += len(_y)

    model = FixedTopicModel(k, vocab_size, alpha, beta)
    for l,t in zip(word_lists, topic_select):
      model.add_new_document(l, debug_m_docs=t)
    flatten = sum(word_lists,[])
    vocab_mat = scipy.sparse.dok_matrix((len(flatten),vocab_size))
    for i, w in enumerate(flatten):
      vocab_mat[i, w] = 1
    assert_mat_equal(<int>model.t.m_vocab, vocab_mat)
    topic_mat = scipy.sparse.dok_matrix((k, k*len(word_lists)))
    for i in xrange(k):
      for j in xrange(len(word_lists)):
        topic_mat[i, j*k+i] = 1
    assert_mat_equal(<int>model.t.m_topic, topic_mat)
    doc_mat = scipy.sparse.dok_matrix((k*len(word_lists), len(flatten)))
    for i, doc in enumerate(word_lists):
      for j, w in enumerate(doc):
        doc_mat[i*k+topic_select[i][j], column_indices[i][j]] = 1
    assert_mat_equal(<int>model.t.m_docs, doc_mat)
    prod1 = np.dot(doc_mat, vocab_mat)
    assert_mat_equal(<int>model.t.view_doc_word.prod, prod1)
    prod2 = np.dot(topic_mat, prod1)
    assert_mat_equal(<int>model.t.view_topic_word.prod, prod2)
    
    exported = model.export_assignment()
    assert_2d_list(exported, topic_select)

    cdef vector* col
    cdef vector* row
    cdef int _i
    for i in xrange(len(word_lists)):
      for j in xrange(len(word_lists[i])):
        _col,_word = model.doc_columns[i][j], word_lists[i][j]
        col = <vector*><int>_col
        row = _get_first(col).row
        matrix_update(model.t.m_docs, -1, row, col)
        doc_mat[i*k+topic_select[i][j], column_indices[i][j]] = 0
        _get_sample_buffer(model.t, col, <double>alpha, <double>beta)
        prod = np.dot(topic_mat, np.dot(doc_mat, vocab_mat))
        for _i in range(k):
          truth = (doc_mat[i*k+_i,:].sum()+alpha)*(prod[_i, _word]+beta)/(prod[_i,:].sum()+vocab_size*beta)
          eq_(<double>truth, <double>model.t.buf[_i].prob)
        matrix_update(model.t.m_docs, +1, row, col)
        doc_mat[i*k+topic_select[i][j], column_indices[i][j]] = 1
    
    for i in xrange(30):
      model.gibbs_iteration()
 
  def test_ln_factorial(self):
    cdef double prod
    for bias in [0.0, 0.3, 0.9, 5.0]:
      for start in [5,10,50]:
        for count in xrange(1, 5):
          prod = 1.0
          for x in xrange(count):
            prod *= (start+x+bias)
          assert_almost_equal(log(prod), ln_factorial(count, start, bias))
    
  def ln_factorial_sum(self, start, count, bias):
    cdef double prod = 0
    cdef int k
    for k in range(count):
      prod += log(bias+k+start)
    return prod
    
    
  def test_hdp(self):
    vocab_size, alpha_table, alpha_topic, beta = 6, 1.0, 1.0, 0.5
    word_lists = [[0,1,2], [3,4,5], [1,3,5], [0,2,4]] 
    initial_table, initial_topic = 3, 3
    s_table = [[random.randint(0,initial_table-1) for x in xrange(len(l))]\
                for l in word_lists]
    s_topic = [[random.randint(0,initial_topic-1) for x in xrange(initial_table)]\
               for l in word_lists]    
    model = HDPTopicModel(vocab_size, alpha_table, alpha_topic, beta,
                          initial_table, initial_topic)
    for i in xrange(len(word_lists)):
      model.add_new_document(word_lists[i], debug_m_docs=s_table[i], \
                             debug_m_topic = s_topic[i])
    flatten = sum(word_lists,[])
    vocab_mat = scipy.sparse.dok_matrix((len(flatten),vocab_size))
    for i, w in enumerate(flatten):
      vocab_mat[i, w] = 1
    assert_mat_equal(<int>model.t.m_vocab, vocab_mat)
    topic_mat = scipy.sparse.dok_matrix((initial_topic, initial_table*len(word_lists)))
    count = 0
    for s_doc in s_topic:
      for topic in s_doc:
        topic_mat[topic, count] = 1
        count += 1
    assert_mat_equal(<int>model.t.m_topic, topic_mat)
    doc_mat = scipy.sparse.dok_matrix((initial_table*len(word_lists), len(flatten)))
    word_count, table_count = 0, 0
    for s_doc in s_table:
      for table in s_doc:
        doc_mat[table_count+table, word_count] = 1
        word_count += 1
      table_count += initial_table
    assert_mat_equal(<int>model.t.m_docs, doc_mat)
    
    prod1 = np.dot(doc_mat, vocab_mat)
    assert_mat_equal(<int>model.t.view_doc_word.prod, prod1)
    prod2 = np.dot(topic_mat, prod1)
    assert_mat_equal(<int>model.t.view_topic_word.prod, prod2)

    cdef vector* col
    cdef vector* row
    cdef vector *row_topic, *col_topic, *row_m_docs, *row_doc_word
    cdef int _i, _t
    word_count, table_count = 0 , 0
    table_prod = np.dot(doc_mat, vocab_mat).tocsr()
      
    model.t.m_topic.squeeze_row = 0

    for i in xrange(len(word_lists)):
      for j in xrange(len(word_lists[i])):
        _col,_word = model.doc_columns[i][j], word_lists[i][j]
        col = <vector*><int>_col
        row = _get_first(col).row
        matrix_update(model.t.m_docs, -1, row, col)
        assert doc_mat[table_count+s_table[i][j], word_count] != 0
        doc_mat[table_count+s_table[i][j], word_count] = 0
        _get_sample_buffer_table(model.t, col, <double>alpha_table, <double>beta)
        prod = np.dot(topic_mat, np.dot(doc_mat, vocab_mat))       
        for _i in range(initial_table):
          _t = s_topic[i][_i]
          truth=doc_mat[table_count+_i,:].sum()*\
                (prod[_t,_word]+beta)/(prod[_t,:].sum()+vocab_size*beta)
          eq_(<double>truth, <double>model.t.buf[_i].prob)
        eq_(alpha_table/vocab_size,  <double>model.t.buf[initial_table].prob)
        matrix_update(model.t.m_docs, +1, row, col)
        doc_mat[table_count+s_table[i][j], word_count] = 1
        word_count += 1       
      prod = np.dot(topic_mat,np.dot(doc_mat, vocab_mat))
      for j in xrange(initial_table):
        _row = model.doc_rows[i][j]
        row_m_docs = <vector*><int>_row
        row_doc_word = mult_view_map_prod_row(model.t.view_doc_word, row_m_docs)
        col_topic = mult_view_map_to_left(model.t.view_topic_word, row_doc_word)
        row_topic = _get_first(col_topic).row       

        matrix_update(model.t.m_topic, -1, row_topic, col_topic)
        assert topic_mat[s_topic[i][j], table_count] != 0
        topic_mat[s_topic[i][j], table_count] = 0

        _get_sample_buffer_topic(model.t, row_doc_word, <double>log(alpha_topic), <double>beta)

        prod = np.dot(topic_mat, np.dot(doc_mat, vocab_mat))
        table_dok = table_prod[table_count, :].todok()        
        for _t in range(initial_topic):
          truth = log(topic_mat[_t, :].sum())
          for _p, _v in table_dok.iteritems():
            _w = _p[1]
            truth += self.ln_factorial_sum(prod[_t, _w], _v, beta)
          truth -= self.ln_factorial_sum(prod[_t, :].sum(), table_dok.sum(), beta*vocab_size)
          if topic_mat[_t,:].sum() == 0:
            assert math.isinf(float(<double>model.t.buf[_t].prob))
          else:
            assert_almost_equal(<double>truth, <double>model.t.buf[_t].prob)
        matrix_update(model.t.m_topic, +1, row_topic, col_topic)
        topic_mat[s_topic[i][j], table_count] = 1        
        table_count += 1

    model.t.m_topic.squeeze_row = 1
    for i in xrange(30):
      model.gibbs_iteration()

