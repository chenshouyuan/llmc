from spmatrix cimport *
from spmatrix import *

cdef class Matrix:
  """ 
  a thin wrapper class of sparse matrix used for rapid prototyping
  """
  cdef matrix *_m
  
  cdef list _row_ptr, _col_ptr
  cdef dict _row_map, _col_map
  cdef bint fixed_row, fixed_col

  # if row/col is fixed, then it is indexed by 1...N
  # otherwise 
  #    the index is the pointer of corresponding vector*
  #    the only guarantee is the different row/col has different 
  #    index

  def __init__(self, matrix=None, fixed=["row", "col"], squeeze=[]):
    self._m = matrix_new(0, 0)
    self._row_ptr = None
    self._col_ptr = None
    if "row" in squeeze:
      self._m.squeeze_row = 1
    if "col" in squeeze:
      self._m.squeeze_col = 1

    self.fixed_row = ("row" in fixed)
    self.fixed_col = ("col" in fixed)

    if not matrix is None:
      self.from_matrix(matrix)
  
  property shape:
    def __get__(self):
      return (self._m.row_count, self._m._col_count)

  property nnz:
    def __get__(self):
      return self._m.nnz

  def from_matrix(self, matrix):
    for x in xrange(matrix.shape[0]):
      matrix_insert_new_row(self._m)
    for x in xrange(matrix.shape[1]):
      matrix_insert_new_col(self._m)    

    if self.fixed_row:
      self._row_ptr = to_data_array(<int>self._m.rows)
      self._row_map = dict([(x,i) for i,x in enumerate(self._row_ptr)])
    if self.fixed_col:
      self._col_ptr = to_data_array(<int>self._m.col)      
      self._col_map = dict([(x,i) for i,x in enumerate(self._col_ptr)])

    for position, value in matrix.iteritems():
      row, col = position
      matrix_update(self._m, <double> value
                    <vector*><int>self._row_ptr[row], 
                    <vector*><int>self._col_ptr[col])

  def to_matrix(self):
    return to_scipy_matrix(<int>self._m)
      
  cdef vector* _get_row(self, int row):
    if self.fixed_row:
      return <vector*><int> self._row_ptr[row]
    else:
      return <vector*><int> row

  cdef vector* _get_col(self, int col):
    if self.fixed_col:
      return <vector*><int> self.right._col_ptr[col]
    else:
      return <vector*><int> col

  def __getitem__(self, tuple item):
    cdef vector* row = self._get_row(item[0])
    cdef vector* col = self._get_col(item[1])
    cdef entry_t value = get_matrix_entry(row, col)
    return value
  
  def __setitem__(self, tuple item, entry_t value):
    cdef vector* row = self._get_row(item[0])
    cdef vector* col = self._get_col(item[1])
    cdef entry_t prev_value = get_matrix_entry(row, col)
    matrix_update(self._m, value-prev_value, row, col)

  cpdef get(self, int row, int col):
    cdef vector* _row = self._get_row(row)
    cdef vector* _col = self._get_col(col)
    cdef entry_t value = get_matrix_entry(_row, _col)
    return value

  cpdef update(self, int row, int col, entry_t delta):
    cdef vector* _row = self._get_row(row)
    cdef vector* _col = self._get_col(col)    
    matrix_update(self._m, delta, _row, _col)

  cpdef getrows(self):
    if self.fixed_row:
      return xrange(len(self._row_ptr))
    else:
      self._row_ptr = to_data_array(<int>self._m.rows)          
      return list(self._row_ptr)

  cpdef getcols(self):
    if self.fixed_col:
      return xrange(len(self._col_ptr))       
    else:
      self._col_ptr = to_data_array(<int>self._m.cols)      
      return list(self._col_ptr)
   
  def listrows(self, col=None):
    cdef list array
    cdef _ll_item *p
    cdef vector *vec
    cdef matrix_entry *entry
    if col is None:
      return self.getrows()      
    else:
      vec = self._get_col(col)
      array = []
      p = vec.store.list.head.next
      while p:
        entry = <matrix_entry*> p.data
        if self.fixed_row:          
          array.append( (self._row_map[<int>entry.row], entry.value) )
        else:
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
      vec = self._get_row(row)
      array = []
      p = vec.store.list.head.next
      while p:
        entry = <matrix_entry*> p.data
        if self.fixed_col:
          array.append( (self._col_map[<int>entry.col], entry.value) )
        else:
          array.append( (<int>entry.col, entry.value) )
        p = p.next
      return array

  def append_row(self, new_row_dict=None):
    assert not self.fixed_row
    cdef vector *row = matrix_insert_new_row(self._m)  
    if not new_row_dict is None:
      for col, value in new_row_dict.iteritems():
        self.update(self._m, <int>row, col, value)
    return <int>row

  def append_col(self, new_col_dict=None):
    assert not self.fixed_col
    cdef vector *col = matrix_insert_new_col(self._m)  
    if not new_col is None:
      for row, value in new_col_dict.iteritems():
        self.update(self._m, row, <int>col, value)
    return <int>col

  def remove_row(self, row):
    assert not self.fixed_row
    matrix_remove_row(self._m, <vector*><int>row)

  def remove_col(self, col):
    assert not self.fixed_col
    matrix_remove_col(self._m, <vector*><int>col)
 
cdef class ProdMatrix(Matrix):
  cdef Matrix left, right
  cdef matrix_mult_view* view

  def __init__(self, left, right):
    self.left, self.right = left, right
    self._m = matrix_new(0, 0)
    self.view = mult_view_new(left._m, right._m, self._m)
    
  cdef vector* _get_row(self, int row):
    cdef vector* _row = <vector*><int> self.left._get_row(row)
    _row = mult_view_map_prod_row(self.view, _row)

  cdef vector* _get_col(self, int col):
    cdef vector* _col = <vector*><int> self.right._get_col(col)
    _col = mult_view_map_prod_col(self.view, _col)
  
  cpdef getrows(self):
    return self.left.getrows()

  cpdef getcols(self):
    return self.right.getcols()

  cpdef to_left(self, row):
    return <int>mult_view_map_to_left(self.view, <vector*><int>row)

  cpdef to_right(self, col):
    return <int>mult_view_map_to_right(self.view, <vector*><int>col)

  # TODO: throw exceptions for append/remove
