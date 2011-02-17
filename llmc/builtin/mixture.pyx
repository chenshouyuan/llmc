from llmc.spmatrix cimport *
from llmc.spmatrix import *
from llmc.cutils cimport *

from libc.stdlib cimport free, malloc, rand, RAND_MAX, memset
from libc.math cimport log, exp
from libc.stdio cimport printf

DEF _DIM = 2
cdef:
  struct data_vec:
    double cord[_DIM]
 
  data_vec* data_vec_new():
    cdef data_vec *vec = <data_vec*>malloc(sizeof(data_vec))
    memset(vec.cord, 0, sizeof(double*_DIM))
    return vec

  void data_vec_free(data_vec *vec):
    free(dm) 

  struct data_matrix:
    femap *rows
    int row_count
  
  data_matrix* data_matrix_new():
    cdef data_matrix *dm = <data_matrix*>malloc(sizeof(data_matrix))
    dm.row_count = 0
    dm.rows = femap_new()
    return dm

  void data_matrix_free(data_matrix *dm):
    femap_delete(dm.rows)
    free(dm)


  void dm_insert_row(data_matrix *dm, data_vec *row):
    dm


  # maintain prod = m*dm 
  struct data_matrix_view:
    matrix *m
    data_matrix *dm, *prod
    femap *col_map, *row_map
  
  data_matrix_view* dm_view_new(matrix *m, data_matrix *dm, data_matrix *prod):
    cdef data_matrix_view *dm_view = <data_matrix_view*>malloc(sizeof(data_matrix_view))
    dm_view.m = m
    dm_view.dm = dm
    dm_view.prod = prod
    dm_view.col_map = femap_new()
    dm_view.row_map = femap_new()
  
  void dm_view_free(data_matrix_view *dm_view):
    femap_delete(dm_view.col_map)
    free(dm_view)

  struct update_callback:
    void *ptr
    void (*update)(void*, int, row_type*, col_type*)
    void (*insert_new_row)(void*, row_type*)
    void (*insert_new_col)(void*, col_type*)
    void (*remove_row)(void*, row_type*)
    void (*remove_col)(void*, col_type*)

  void cb_dm_update(void *ptr, int delta, row_type *row, col_type *col):
    cdef data_matrix_view *dm_view = <data_matrix_view*> ptr
    data_vec *prod_row = femap_lookup(dm_view.row_map, row)
    data_vec *dm_row = femap_lookup(dm_view.col_map, col)
    cdef int i
    for i in range(_DIM):
      prod_row.cord[i] += delta*dm_row.cord[i]
  
  void cb_dm_insert_new_row(void *ptr, row_type* row):
    cdef data_matrix_view *dm_view = <data_matrix_view*> ptr
    data_vec *prod_row = data_vec_new()
    femap_insert(dm_view, <void*>row, <void*>prod_row)

  void cb_dm_insert_new_col(void *ptr, col_type* col):
    cdef data_matrix_view *dm_view = <data_matrix_view*> ptr
    data_vec *dm_row = data_vec_new()
