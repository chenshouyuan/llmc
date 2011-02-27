from spmatrix cimport *
from spmatrix import *

cdef class Matrix:
  """ 
  a wrapper class of sparse matrix used for rapid prototyping
  """
  cdef matrix *_m
  
  cdef list _row_ptr, _col_ptr
  cdef dict _row_map, _col_map

  def __init__(self, matrix=None, squeeze=[]):
    self._m = matrix_new(0, 0)
    self._row_ptr = None
    self._col_ptr = None
    if "row" in squeeze:
      self._m.squeeze_row = 1
    if "col" in squeeze:
      self._m.squeeze_col = 1
    if not matrix is None:
      self.from_matrix(matrix)
  
  property shape:
    def __get__(self):
      return (len(self._row_ptr), len(self._col_ptr))

  property nnz:
    def __get__(self):
      return self._m.nnz

  cdef vector* _get_row(self, int row):
    cdef vector* _row = <vector*><int> self._row_ptr[row]
    return _row

  cdef vector* _get_col(self, int col):
    cdef vector* _col = <vector*><int> self.right._col_ptr[col]
    return _row

  # users should make sure that the target row/col still exists
  # use getrows/getcols after structrcal changes
  def __getitem__(self, tuple item):
    cdef vector* row = self._get_row(item[0])
    cdef vector* col = self._get_col(item[1])
    cdef entry_t value = get_matrix_entry(row, col)
    return value
  
  # users should make sure that the target row/col still exists
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
    self._row_ptr = to_data_array(<int>self._m.rows)
    self._row_map = dict([(x,i) for i,x in enumerate(self._row_ptr)])
    return xrange(len(self._row_ptr))

  cpdef getcols(self):
    self._col_ptr = to_data_array(<int>self._m.cols)
    self._col_map = dict([(x,i) for i,x in enumerate(self._col_ptr)])
    return xrange(len(self._col_ptr))

  def from_matrix(self, matrix):
    for x in xrange(matrix.shape[0]):
      matrix_insert_new_row(self._m)
    for x in xrange(matrix.shape[1]):
      matrix_insert_new_col(self._m)    
    self.getrows()
    self.getcols()
    for position, value in matrix.iteritems():
      row, col = position
      self.update(row,col,value)

  def to_matrix(self):
    return to_scipy_matrix(<int>self._m)

  def squeeze(self):
    self.getrows()
    self.getcols()

  def listrows(self, col=None):
    cdef list array
    cdef _ll_item *p
    cdef vector *vec
    cdef matrix_entry *entry
    if col is None:
      return self.getrows()      
    else:
      vec = <vector*><int>self._col_ptr[col] 
      array = []
      p = vec.store.list.head.next
      while p:
        entry = <matrix_entry*> p.data
        array.append( (self._row_map[<int>entry.row], entry.value) )
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
      vec = <vector*><int>self._row_ptr[row] 
      array = []
      p = vec.store.list.head.next
      while p:
        entry = <matrix_entry*> p.data
        array.append( (self._col_map[<int>entry.col], entry.value) )
        p = p.next
      return array

  def append_row(self, new_row=None):
    cdef vector *row = matrix_insert_new_row(self._m)  
    cdef int row_id = len(self._row_ptr)
    self._row_map[<int>row] = row_id
    self._row_ptr.append(<int>row)    
    if not new_row is None:
      for position, value in new_row.iteritems():
        __row1,col = position
        self.update(row_id, col, value)
    return row_id

  def append_col(self, new_col=None):
    cdef vector *col = matrix_insert_new_col(self._m)  
    cdef int col_id = len(self._col_ptr)
    self._col_map[<int>col] = col_id
    self._col_ptr.append(<int>col)    
    if not new_col is None:
      for position, value in new_col.iteritems():
        row ,__col1 = position
        self.update(row, col_id, value)    
    return col_id

  # structure change! 
  # must call getrows() to refresh before updating values
  def remove_row(self, row):
    matrix_remove_row(self._m, <vector*><int>self._row_ptr[row])
    self._row_map.pop(self._row_ptr[row])

  # structure change! 
  # must call getcols() to refresh before updating values
  def remove_col(self, col):
    matrix_remove_col(self._m, <vector*><int>self._col_ptr[col])
    self._col_map.pop(self._col_ptr[col])
 
cdef class ProdMatrix(Matrix):
  cdef Matrix left, right
  cdef matrix_mult_view* view

  def __init__(self, left, right):
    self.left, self.right = left, right
    self._m = matrix_new(0, 0)
    self.view = mult_view_new(left._m, right._m, self._m)
    
  cdef vector* _get_row(self, int row):
    cdef vector* _row = <vector*><int> self.left._row_ptr[row]
    _row = mult_view_map_prod_row(self.view, _row)

  cdef vector* _get_col(self, int col):
    cdef vector* _col = <vector*><int> self.right._col_ptr[col]
    _col = mult_view_map_prod_col(self.view, _col)

  # TODO: throw exceptions for append/remove
