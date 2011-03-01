from spmatrix cimport *
from spmatrix import *

cdef class Matrix:
  """ 
  a thin wrapper class of sparse matrix used for rapid prototyping
  """
  cdef matrix *_m
  
  #    the index is the pointer of corresponding vector*
  #    the only guarantee is the different row/col has different 
  #    index

  def __init__(self, matrix=None, squeeze=[]):
    self._m = matrix_new(0, 0)
    if "row" in squeeze:
      self._m.squeeze_row = 1
    if "col" in squeeze:
      self._m.squeeze_col = 1
    if not matrix is None:
      self.from_matrix(matrix)
  
  property shape:
    def __get__(self):
      return (self._m.row_count, self._m.col_count)
  property nnz:
    def __get__(self):
      return self._m.nnz

  def from_matrix(self, matrix):
    self.set_rows(matrix.shape[0])
    self.set_cols(matrix.shape[1])
    self.set_matrix(matrix)    
  
  def set_rows(self, int row_count):
    cdef int x
    for x in range(row_count):
      matrix_insert_new_row(self._m)

  def set_cols(self, int col_count):
    cdef int x
    for x in range(col_count):
      matrix_insert_new_col(self._m)    

  def set_matrix(self, matrix):
    rows = to_data_array(<int>self._m.rows)
    cols = to_data_array(<int>self._m.cols)
    for position, value in matrix.iteritems():
      row, col = position
      matrix_update(self._m, <double> value,                    
                    <vector*><int>rows[row], 
                    <vector*><int>cols[col])

  def to_matrix(self):
    return to_scipy_matrix(<int>self._m)
      
  def __getitem__(self, tuple item):
    cdef vector* row = <vector*><int>item[0]
    cdef vector* col = <vector*><int>item[1]
    cdef entry_t value = get_matrix_entry(row, col)
    return value
  
  def __setitem__(self, tuple item, entry_t value):
    cdef vector* row = <vector*><int>item[0]
    cdef vector* col = <vector*><int>item[1]    
    cdef entry_t prev_value = get_matrix_entry(row, col)
    matrix_update(self._m, value-prev_value, row, col)

  cpdef get(self, int row, int col):
    cdef entry_t value = get_matrix_entry(<vector*><int>row, <vector*><int>col)
    return value

  cpdef update(self, int row, int col, entry_t delta):
    matrix_update(self._m, delta, <vector*><int>row, <vector*><int>col)

  cpdef getrows(self):
    return to_data_array(<int>self._m.rows)

  cpdef getcols(self):
    return to_data_array(<int>self._m.cols)    
   
  def listrows(self, col=None):
    cdef list array
    cdef _ll_item *p
    cdef vector *vec
    cdef matrix_entry *entry
    if col is None:
      return self.getrows()      
    else:
      vec = <vector*><int> col
      array = []
      p = vec.store.list.head.next
      while p:
        entry = <matrix_entry*> p.data
        array.append( (<int>entry.row, entry.value) )
        p = p.next
      return array

  def listcols(self, row=None):
    cdef list array
    cdef _ll_item *p
    cdef vector *vec
    cdef matrix_entry *entry
    if row is None:
      return self.getcols()      
    else:
      vec = <vector*><int> row
      array = []
      p = vec.store.list.head.next
      while p:
        entry = <matrix_entry*> p.data
        array.append( (<int>entry.col, entry.value) )
        p = p.next
      return array

  def append_row(self, new_row_dict=None):
    cdef vector *row = matrix_insert_new_row(self._m)  
    if not new_row_dict is None:
      for col, value in new_row_dict.iteritems():
        self.update(<int>row, col, value)
    return <int>row

  def append_col(self, new_col_dict=None):
    cdef vector *col = matrix_insert_new_col(self._m)  
    if not new_col_dict is None:
      for row, value in new_col_dict.iteritems():
        self.update(row, <int>col, value)
    return <int>col

  def remove_row(self, row):
    matrix_remove_row(self._m, <vector*><int>row)

  def remove_col(self, col):
    matrix_remove_col(self._m, <vector*><int>col)
 
cdef class ProdMatrix(Matrix):
  cdef Matrix left, right
  cdef matrix_mult_view* view

  def __init__(self, Matrix left, Matrix right):
    self.left, self.right = left, right
    self._m = matrix_new(0, 0)
    self.view = mult_view_new(left._m, right._m, self._m)

  cpdef left_col(self, row):
    return <int>mult_view_map_to_left(self.view, <vector*><int>row)

  cpdef right_row(self, col):
    return <int>mult_view_map_to_right(self.view, <vector*><int>col)

  cpdef prod_row(self, row):
    return <int>mult_view_map_prod_row(self.view, <vector*><int>row)

  cpdef prod_col(self, col):
    return <int>mult_view_map_prod_col(self.view, <vector*><int>col)
