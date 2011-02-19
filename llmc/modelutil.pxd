from spmatrix cimport *
from libc.math cimport log, exp
from libc.stdlib cimport rand, RAND_MAX

cdef inline matrix_entry* _get_first(vector* vec):
  return <matrix_entry*>vec.store.list.head.next.data

cdef:
  struct sample_buffer:
    double prob
    void  *ptr

  inline void* sample_unnormalized(sample_buffer *buf, int count):
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

  inline double _log_sum_normalize(sample_buffer *buf, int count):
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

  inline void* sample_log_unnormalized(sample_buffer *buf, int count):
    cdef double log_norm = _log_sum_normalize(buf, count)
    cdef int i 
    for i in range(count):
      buf[i].prob = exp(buf[i].prob)
    return sample_unnormalized(buf, count)

