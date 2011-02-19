# hash table implementation from hash-table.h
cdef struct HashTable

ctypedef void *HashTableKey
ctypedef void *HashTableValue

cdef:
  struct _ll_item:
    void* data
    _ll_item *prev, *next
  struct _linked_list:
    _ll_item *head, *tail

# fast enumerable hash map
cdef:
  struct femap:
    HashTable *table
    _linked_list *list 
  
  HashTableValue femap_lookup(femap* f, HashTableKey key)  
  void femap_insert(femap* f, HashTableKey key, HashTableValue value)
  void femap_remove(femap* f, HashTableKey key)
  femap* femap_new()
  void femap_delete(femap* f)

ctypedef double entry_t
# matrix structures
cdef:
  struct vector:
    entry_t sum
    femap *store
    _ll_item *link      
    vector* parent # !=0 if is a vector in product matrix
    void *aux # customizable data field
  
  ctypedef vector col_type
  ctypedef vector row_type  
  
  struct matrix_entry:
    entry_t value  
    vector *row, *col  

  matrix_entry* matrix_entry_new(entry_t value, row_type *row, col_type *col)
  void matrix_entry_delete(matrix_entry *entry)
  entry_t get_matrix_entry(row_type *row, col_type *col)
  int update_matrix_entry(entry_t delta, row_type *row, col_type *col)

  struct update_callback

  struct matrix:
    _linked_list *rows, *cols    
    int row_count, col_count, nnz
    bint squeeze_row, squeeze_col
    int callback_count
    update_callback *callbacks[10]
  
  matrix* matrix_new(bint squeeze_row, bint squeeze_col)
  void matrix_delete(matrix *m) 

  vector* matrix_insert_new_row(matrix *m)
  vector* matrix_insert_new_col(matrix *m)
  void matrix_remove_row(matrix *m, row_type *row)
  void matrix_remove_col(matrix *m, col_type *col)  

cdef void matrix_update(matrix* m, entry_t delta, row_type* row, col_type *col)

# incremental maintainence of matrix multiplication
cdef:
  struct matrix_mult_view:
    matrix *left, *right, *prod
    HashTable *right_row_map, *left_col_map
    HashTable *right_col_map, *left_row_map
  
  vector* mult_view_map_to_left(matrix_mult_view *view, row_type *row)
  vector* mult_view_map_to_right(matrix_mult_view *view, col_type *col)
  vector* mult_view_map_prod_row(matrix_mult_view *view, row_type *row)
  vector* mult_view_map_prod_col(matrix_mult_view *view, col_type *col)

  matrix_mult_view* mult_view_new(matrix *left, matrix *right, matrix *prod)
  void mult_view_delete(matrix_mult_view* view)


cdef inline matrix_entry* _get_first(vector* vec):
  return <matrix_entry*>vec.store.list.head.next.data

