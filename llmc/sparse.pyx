"""
  Sparse Matrix Utiltities
    1) _NO_ random access
    2) enumerating over rows and columns
    3) view maintainence on row and columns
    4) view maintainence on product of matrix
    5) add/remove rows and columns
    6) squeeze out empty rows

"""
from libc.stdlib cimport free, malloc, rand, RAND_MAX
from libc.stdio cimport printf

cdef extern from 'hash-table.h':
  struct HashTable:
    pass
  ctypedef void *HashTableKey
  ctypedef void *HashTableValue
  ctypedef unsigned int (*HashTableHashFunc)(HashTableKey value)
  ctypedef int (*HashTableEqualFunc)(HashTableKey value1, HashTableKey value2)
  HashTable *hash_table_new(HashTableHashFunc hash_func, 
                            HashTableEqualFunc equal_func)
  void hash_table_free(HashTable *hash_table)
  int hash_table_insert(HashTable *hash_table, 
                        HashTableKey key, 
                        HashTableValue value)
  HashTableValue hash_table_lookup(HashTable *hash_table, 
                                   HashTableKey key)
  int hash_table_remove(HashTable *hash_table, HashTableKey key)
 
cdef:
  unsigned int pointer_hash(void *location):
    return <unsigned int><unsigned long>location
  int pointer_equal(void *l1, void*l2):
    return 1 if l1==l2 else 0

# utility data structures
cdef:
  # array lists
  struct array_list:
    int   count
    void* items  

  void al_initialize(array_list* l, int count, int item_size):
    l.count = count
    l.items = malloc(count*item_size) 

  void al_delete(array_list* l):
    free(l.items)
    free(l)
  
  # linked list
  struct _ll_item:
    void* data
    _ll_item *prev, *next

  struct _linked_list:
    _ll_item *head, *tail
  
  _ll_item* ll_append(_linked_list *list, void* data):
    cdef _ll_item *new = <_ll_item*>malloc(sizeof(_ll_item))
    new.data = data
    new.prev = list.tail
    list.tail.next = new
    new.next = <_ll_item*>0
    list.tail = new
    return new

  void ll_remove(_linked_list *list, _ll_item *item):
    item.prev.next = item.next
    if item == list.tail:
      list.tail = item.prev
    else:
      item.next.prev = item.prev      
    free(item)
  
  _linked_list* ll_new():
    cdef _linked_list *new=<_linked_list*>malloc(sizeof(_linked_list))
    new.head = <_ll_item*>malloc(sizeof(_ll_item))
    new.head.data = <void*>0
    new.head.prev = <_ll_item*>0 
    new.head.next = <_ll_item*>0
    new.tail = new.head
    return new

  void ll_delete(_linked_list *list):
    cdef _ll_item *p = list.head.next
    cdef _ll_item *q
    while p:
      q = p.next
      free(p)
      p = q
    free(list.head)
    free(list)

  ctypedef void(*data_func)(void*)

  void ll_for_each(_linked_list *list, data_func func):
    cdef _ll_item *p = list.head.next
    cdef _ll_item *q
    while p:
      q = p.next
      func(p.data)
      p = q

# Fast enumerable hash map
cdef:
  struct femap:
    # table: data -> _ll_item(data)
    HashTable *table
    _linked_list *list 
 
  HashTableValue femap_lookup(femap* f, HashTableKey key):
    cdef HashTableValue v = hash_table_lookup(f.table, key)
    if v:
      v = (<_ll_item*>v).data
    return v
  
  void femap_insert(femap* f, HashTableKey key, HashTableValue value):
    cdef _ll_item *new
    if not hash_table_lookup(f.table, key):
      new = ll_append(f.list, value)      
      hash_table_insert(f.table, key, <void*>new)

  void femap_remove(femap* f, HashTableKey key):
    cdef _ll_item* item=<_ll_item*>hash_table_lookup(f.table,key)
    if item:
      ll_remove(f.list,item)
      hash_table_remove(f.table,key)

  void femap_for_each(femap* f, data_func func):
    ll_for_each(f.list, func)
      
  femap* femap_new():
    cdef femap* new = <femap*>malloc(sizeof(femap))
    new.table = hash_table_new(pointer_hash, pointer_equal)
    new.list = ll_new()
    return new

  void femap_delete(femap* f):
    ll_delete(f.list)
    hash_table_free(f.table)
    free(f) 

# matrix data structure
cdef:
  struct vector:
    int sum
    femap *store
    _ll_item *link      
    vector* parent # !=0 if is a vector in product matrix
    void *aux # customizable data field

  vector* vector_new():
    cdef vector* v = <vector*>malloc(sizeof(vector))
    v.sum = 0
    v.store = femap_new()
    v.parent = <vector*> 0
    v.aux = <void*> 0
    return v

  void vector_delete(vector *v):
    femap_delete(v.store)
    free(v)

  ctypedef vector col_type
  ctypedef vector row_type
  
  struct matrix_entry:
    int value  
    vector *row, *col
    
  matrix_entry* matrix_entry_new(int value,
             row_type *row, col_type *col):   
    cdef matrix_entry* new = <matrix_entry*>malloc(sizeof(matrix_entry))
    new.value = value
    new.row = row
    new.col = col
    femap_insert(row.store, <void*>col, <void*>new)
    femap_insert(col.store, <void*>row, <void*>new)
    return new

  void matrix_entry_delete(matrix_entry *entry):
    femap_remove(entry.row.store, <void*>entry.col)
    femap_remove(entry.col.store, <void*>entry.row)
    free(entry)

  int get_matrix_entry(row_type *row, col_type *col):
    cdef matrix_entry *p = <matrix_entry*>femap_lookup(col.store, <void*>row)
    if not p:
      return 0
    else:
      return p.value 

  int update_matrix_entry(int delta, row_type *row, col_type *col):
    row.sum += delta
    col.sum += delta   
    cdef matrix_entry *p = \
      <matrix_entry*>femap_lookup(col.store, <void*>row)
    if p:
      p.value += delta
      if p.value == 0:
        matrix_entry_delete(p)
        return -1
      return 0
    else:
      p = matrix_entry_new(delta, row, col)
      return 1
   
  struct update_callback:
    void *ptr
    void (*update)(void*, int, row_type*, col_type*)
    void (*insert_new_row)(void*, row_type*)
    void (*insert_new_col)(void*, col_type*)
    void (*remove_row)(void*, row_type*)
    void (*remove_col)(void*, col_type*)

  struct matrix:
    _linked_list *rows, *cols    
    int row_count, col_count, nnz
    bint squeeze_row, squeeze_col
    int callback_count
    update_callback *callbacks[10]

  vector* matrix_insert_new_row_dry(matrix *m):
    cdef vector* new_row = vector_new()
    m.row_count += 1    
    new_row.link = ll_append(m.rows, <void*>new_row)
    return new_row

  vector* matrix_insert_new_col_dry(matrix *m):
    cdef vector* new_col = vector_new()    
    m.col_count += 1    
    new_col.link = ll_append(m.cols, <void*>new_col)
    return new_col  

  vector* matrix_insert_new_row(matrix *m):
    cdef vector* new_row = matrix_insert_new_row_dry(m)
    cdef int i
    for i in range(m.callback_count):
      if m.callbacks[i].insert_new_row:
        m.callbacks[i].insert_new_row(m.callbacks[i].ptr, new_row)
    return new_row

  vector* matrix_insert_new_col(matrix *m):
    cdef vector* new_col = matrix_insert_new_col_dry(m)
    cdef int i
    for i in range(m.callback_count):
      if m.callbacks[i].insert_new_col:
        m.callbacks[i].insert_new_col(m.callbacks[i].ptr, new_col)
    return new_col

  void matrix_register_callback(matrix *m, update_callback *cb):
    m.callbacks[m.callback_count] = cb
    m.callback_count += 1


  matrix* matrix_new_dry():
    cdef matrix* m = <matrix*>malloc(sizeof(matrix))    
    m.row_count = 0
    m.col_count = 0
    m.squeeze_row = 0
    m.squeeze_col = 0
    m.callback_count = 0 
    return m

  matrix* matrix_new(bint squeeze_row, bint squeeze_col):
    cdef matrix* m = matrix_new_dry()
    m.rows = ll_new()
    m.cols = ll_new()
    m.squeeze_row = squeeze_row
    m.squeeze_col = squeeze_col
    return m

  void _vector_delete_iter(void* data):
    cdef vector* v = <vector*>data
    vector_delete(v)

  void matrix_delete(matrix *m):
    cdef int i
    ll_for_each(m.rows, _vector_delete_iter)
    ll_for_each(m.cols, _vector_delete_iter)
    ll_delete(m.rows)
    ll_delete(m.cols)
    for i in range(m.callback_count):
      free(m.callbacks[i])
    free(m)

  void matrix_remove_row(matrix *m, row_type *row):
    cdef int i
    for i in range(m.callback_count):
      if m.callbacks[i].remove_row:
        m.callbacks[i].remove_row(m.callbacks[i].ptr, row)    
    m.row_count -= 1
    vector_delete(row)    
    ll_remove(m.rows, row.link)

  void matrix_remove_col(matrix *m, col_type *col):
    cdef int i
    for i in range(m.callback_count):
      if m.callbacks[i].remove_col:
        m.callbacks[i].remove_col(m.callbacks[i].ptr, col)       
    m.col_count -= 1
    vector_delete(col)    
    ll_remove(m.cols, col.link)

  void matrix_update(matrix* m, int delta, row_type* row, col_type *col):
    cdef int res = update_matrix_entry(delta, row, col)
    cdef int i
    m.nnz += res
    for i in range(m.callback_count):
      if m.callbacks[i].update:
        m.callbacks[i].update(m.callbacks[i].ptr, delta, row, col)
    if m.squeeze_row and row.sum == 0:
      matrix_remove_row(m, row)
    if m.squeeze_col and col.sum == 0:
      matrix_remove_col(m, col)  

  struct matrix_mult_view:
    matrix *left, *right, *prod
    HashTable *right_row_map, *left_col_map
    HashTable *right_col_map, *left_row_map

  vector* mult_view_map_to_left(matrix_mult_view *view, row_type *row):
    return <vector*>hash_table_lookup(view.right_row_map, row)

  vector* mult_view_map_to_right(matrix_mult_view *view, col_type *col):
    return <vector*>hash_table_lookup(view.left_col_map, col)

  vector* mult_view_map_prod_row(matrix_mult_view *view, row_type *row):
    return <vector*>hash_table_lookup(view.left_row_map, row)

  vector* mult_view_map_prod_col(matrix_mult_view *view, col_type *col):
    return <vector*>hash_table_lookup(view.right_col_map, col)

  void cb_mult_update_right(void *ptr, int delta, row_type *row, col_type *col):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    cdef vector* prod_row      
    cdef vector* prod_col = <vector*>hash_table_lookup(view.right_col_map, <void*>col)
    cdef vector* col_left = <vector*>hash_table_lookup(view.right_row_map, <void*>row)
    cdef _ll_item *p = col_left.store.list.head.next    
    cdef matrix_entry* entry 
    while p:
      entry = <matrix_entry*> p.data
      prod_row = <vector*>hash_table_lookup(view.left_row_map, <void*>entry.row)
      matrix_update(view.prod, delta*entry.value, prod_row, prod_col)
      p = p.next

  void cb_mult_update_left(void *ptr, int delta, row_type *row, col_type *col):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    cdef vector* prod_col 
    cdef vector* prod_row = <vector*>hash_table_lookup(view.left_row_map, <void*>row)        
    cdef vector* row_right = <vector*>hash_table_lookup(view.left_col_map, <void*>col)
    cdef _ll_item *p = row_right.store.list.head.next    
    cdef matrix_entry* entry 
    while p:
      entry = <matrix_entry*> p.data
      prod_col = <vector*>hash_table_lookup(view.right_col_map, <void*>entry.col)
      matrix_update(view.prod, delta*entry.value, prod_row, prod_col)
      p = p.next

  void cb_mult_remove_col_right(void *ptr, col_type *col):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    cdef vector* prod_col = <vector*>hash_table_lookup(view.right_col_map, <void*>col)    
    matrix_remove_col(view.prod, prod_col)
    hash_table_remove(view.right_col_map, col)
  
  void cb_mult_remove_row_left(void *ptr, row_type *row):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    cdef vector* prod_row = <vector*>hash_table_lookup(view.left_row_map, <void*>row)    
    matrix_remove_row(view.prod, prod_row) 
    hash_table_remove(view.left_row_map, row)   

  void cb_mult_remove_col_left(void *ptr, col_type *col):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    if not hash_table_lookup(view.left_col_map, <void*>col):
      return                    
    cdef vector* right_row = <vector*>hash_table_lookup(view.left_col_map, <void*>col)    
    hash_table_remove(view.left_col_map, col)
    hash_table_remove(view.right_row_map, right_row)    
    matrix_remove_row(view.right, right_row)

  void cb_mult_remove_row_right(void *ptr, row_type *row):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    if not hash_table_lookup(view.right_row_map, row):
      return    
    cdef vector* left_col = <vector*>hash_table_lookup(view.right_row_map, <void*>row)    
    hash_table_remove(view.right_row_map, row)   
    hash_table_remove(view.left_col_map, left_col)            
    matrix_remove_col(view.left, left_col)
 
  void cb_mult_insert_new_row_left(void *ptr, row_type *row):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    cdef vector* new_row = matrix_insert_new_row(view.prod)
    new_row.parent = row
    hash_table_insert(view.left_row_map, <void*>row, <void*>new_row)

  void cb_mult_insert_new_col_right(void *ptr, col_type *col):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    cdef vector* new_col = matrix_insert_new_col(view.prod)
    new_col.parent = col
    hash_table_insert(view.right_col_map, <void*>col, <void*>new_col)

  void cb_mult_insert_new_col_left(void *ptr, col_type *col):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    if hash_table_lookup(view.left_col_map, <void*>col):
      return                
    cdef vector* new_row = matrix_insert_new_row_dry(view.right)    
    for i in range(view.right.callback_count):
      if view.right.callbacks[i].ptr != ptr:
        view.right.callbacks[i].insert_new_row(\
            view.right.callbacks[i].ptr, new_row)
    hash_table_insert(view.left_col_map, <void*>col, <void*>new_row)
    hash_table_insert(view.right_row_map, <void*>new_row, <void*>col)   

  void cb_mult_insert_new_row_right(void *ptr, row_type *row):
    cdef matrix_mult_view *view = <matrix_mult_view*> ptr
    if hash_table_lookup(view.right_row_map, row):
      return    
    cdef vector* new_col = matrix_insert_new_col_dry(view.left)    
    for i in range(view.left.callback_count):
      if view.left.callbacks[i].ptr != ptr:
        view.left.callbacks[i].insert_new_col(\
            view.left.callbacks[i].ptr, new_col)    
    hash_table_insert(view.left_col_map, <void*>new_col, <void*>row)    
    hash_table_insert(view.right_row_map, <void*>row, <void*>new_col)    


  # associate three matrices, left, right and prod
  # such that left*right = prod is always maintained
  # NOTE: this function requires left/right/prod being empty

  matrix_mult_view* mult_view_new(matrix *left, matrix *right, matrix *prod):
    cdef matrix_mult_view *view = <matrix_mult_view*>malloc(sizeof(matrix_mult_view))
    cdef update_callback *cb

    view.left = left
    view.right = right
    view.prod = prod

    view.left_col_map = hash_table_new(pointer_hash, pointer_equal)
    view.left_row_map = hash_table_new(pointer_hash, pointer_equal)
    view.right_col_map = hash_table_new(pointer_hash, pointer_equal)
    view.right_row_map = hash_table_new(pointer_hash, pointer_equal)

    cb = <update_callback*>malloc(sizeof(update_callback))
    cb.update = cb_mult_update_left
    cb.insert_new_row = cb_mult_insert_new_row_left
    cb.insert_new_col = cb_mult_insert_new_col_left
    cb.remove_row = cb_mult_remove_row_left
    cb.remove_col = cb_mult_remove_col_left
    cb.ptr = <void*>view
    matrix_register_callback(view.left, cb)
    
    cb = <update_callback*>malloc(sizeof(update_callback))
    cb.update = cb_mult_update_right
    cb.insert_new_row = cb_mult_insert_new_row_right
    cb.insert_new_col = cb_mult_insert_new_col_right
    cb.remove_row = cb_mult_remove_row_right
    cb.remove_col = cb_mult_remove_col_right
    cb.ptr = <void*>view
    matrix_register_callback(view.right, cb)  
    return view

  void mult_view_delete(matrix_mult_view* view):
    hash_table_free(view.left_col_map)
    hash_table_free(view.left_row_map)
    hash_table_free(view.right_col_map)
    hash_table_free(view.right_row_map)
    free(view)
 
  # summation is not useful 
  # when the matrix structure varies overtime (HDP)
  # TODO: check this again

##############
# lda
##############

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

  inline vector* _to_l_1(matrix_mult_view *v1, vector* vec):
    cdef vector* temp
    temp = mult_view_map_prod_col(v1, vec)
    return temp

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
    t.buf[count].ptr = 0
    count += 1
    return count
  
  # for topic assignment in HDP
  int _get_sample_buffe_topic(topic_model *t, vector *col, double log_alpha_topic, double beta):
    cdef vector *word, *topic_row, *row
    cdef _ll_item *p
    cdef int count = 0
    p = t.m_topic.rows.list.head.next
    while p:
      row = <vector*> p.data
      topic_row = _to_l_1(t.view_topic_word, row)
      t.buf[count].prob = log(row.sum) + _log_posterior_table(topic_row, col, beta)
      t.buf[count].ptr = <void*> row
      count += 1
      p = p.next
    t.buf[count].prob = log_alpha_topic + _log_prior_table(col, beta)
    t.buf[count].ptr = 0
    count += 1
    return count

  double _log_prior_table(vector *table, double beta):
    pass
    
  double _log_posterior_table(vector *topic, vector *table, double beta):
    pass

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
  
  # gibbs iteration of HDP
  void resample_hdp(topic_model *t, double alpha_topic, double alpha_topic, double beta):
    cdef _ll_item *p = t.m_docs.cols.head.next
    
    
import random

cdef class FixedTopicModel:
  cdef topic_model *t
  cdef object _vocab_list
  cdef object _topic_rows
  cdef int    _topic_count
  cdef double alpha, beta
  cdef object doc_columns

  def __init__(self, k, vocab_size, alpha, beta):
    cdef matrix *prod1, *prod2
    self.alpha = alpha
    self.beta = beta

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

    if not k is None:
      self.set_topic_num(k)    
    if not vocab_size is None:
      self.set_vocab_size(vocab_size)

    self.doc_columns = []

  def set_topic_num(self, k):
    self._topic_count = <int> k
    self._topic_rows = [<int>matrix_insert_new_row(self.t.m_topic) for i in xrange(k)]

  def set_vocab_size(self, vocab_size):
    self.t.vocab_size = vocab_size
    self._vocab_list = [<int>matrix_insert_new_col(self.t.m_vocab) for i in xrange(vocab_size)]

  def add_new_document(self, word_list, debug=None):
    cdef vector *topic_row, *row, *col, *temp
    cdef vec_group *group = <vec_group*>malloc(sizeof(vec_group))

    self.t.doc_count += 1

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
  
    word_cols = [<int>matrix_insert_new_col(self.t.m_docs) for w in word_list]
    j = 0
    for _col, word in zip(word_cols, word_list):
      col = <vector*><int> _col
      row = mult_view_map_to_right(self.t.view_doc_word, col)
      temp = <vector*><int> self._vocab_list[word]
      matrix_update(self.t.m_vocab, +1, row, temp)          
      col.aux = <void*> group             
      if debug:
        _row = topics[debug[j]]
      else:
        _row = random.choice(topics)
      row = <vector*><int> _row
      matrix_update(self.t.m_docs, +1, row, col)    
      j += 1
    self.doc_columns.append(word_cols)

  def gibbs_iteration(self):
    resample_topic_model(self.t, self.alpha, self.beta)

  def export_assignment(self):
    sparse = _to_scipy_matrix(<int>self.t.m_docs)
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


cdef:
    pass


###################### 
# unit testing       #  
######################
from nose.tools import *
import unittest

def to_array(_l):
  l = <_linked_list*><int> _l
  p = l.head.next
  array = []
  while p:
    q = <_ll_item*><int> p
    array.append(<int>p)
    p = p.next
  return array

def to_data_array(_l):
  l = <_linked_list*><int> _l
  p = l.head.next
  array = []
  while p:
    q = <_ll_item*><int> p
    array.append(<int>p.data)
    p = p.next
  return array


class TestLinkList:
  def test_create_destroy(self):
    l = ll_new()
    ll_delete(l)  

  def check_equality(self, pointers, array):
    eq_(len(pointers), len(array))
    for _p, a in zip(pointers, array):
      p = <_ll_item*><int> _p
      assert p.data==<void*><int> a

  def to_array(self, _l):
    return to_array(_l)

  def test_append_remove(self):
    l = ll_new()
    array = [1,2,3,4,5]
    for x in array:
      ll_append(l, <void*><int>x)
    pointers = self.to_array(<int>l)   
    self.check_equality(pointers, array)
    # remove the last
    p = pointers.pop()
    ll_remove(l, <_ll_item*><int>p)
    array.pop()
    pointers1 = self.to_array(<int>l)   
    for p,q in zip(pointers, pointers1):
      assert p==q
    self.check_equality(pointers1, array)
    # remove the second
    p = pointers.pop(1)
    ll_remove(l, <_ll_item*><int>p)
    array.pop(1)
    self.check_equality(self.to_array(<int>l), array)
    # append another
    array.append(6)
    ll_append(l, <void*><int>6)
    self.check_equality(self.to_array(<int>l), array)
    ll_delete(l) 

class TestFemap:
  def test_create_delete(self):
    l = femap_new()
    femap_delete(l)  
  
  def test_add_remove(self):
    f = femap_new()
    example = {1:2, 2:4, 3:8, 4:16}
    for key in example:
      value = example[key]
      femap_insert(f, <void*><int>key, <void*><int>value)
    self.check_equality(<int>f, example)   
    femap_remove(f, <void*>1)
    example.pop(1)
    self.check_equality(<int>f, example)
    femap_remove(f, <void*>3)
    example.pop(3)
    self.check_equality(<int>f, example)
    femap_insert(f, <void*>4, <void*>5)
      # should not modifiy f
    self.check_equality(<int>f, example)
    femap_insert(f, <void*>5, <void*>5)
    example[5] = 5
    self.check_equality(<int>f, example)
    femap_insert(f, <void*>6, <void*>6)
    example[6] = 6
    self.check_equality(<int>f, example)
    femap_delete(f)

  def check_equality(self, _f, dic):
    f = <femap*><int>_f
    for key in dic:
      res = femap_lookup(f, <void*><int>key)
      assert res == <void*><int>dic[key]
    value_set = set()
    p = f.list.head.next
    while p:
      value_set.add(<int>p.data)
      p = p.next
    assert value_set==set(dic.values())

cpdef _set_matrix(_m, mat):
  cdef matrix* m = <matrix*><int>_m
  row_p = [<int>matrix_insert_new_row(m) for i in xrange(mat.shape[0])]
  col_p = [<int>matrix_insert_new_col(m) for i in xrange(mat.shape[1])]
  for key, value in mat.iteritems():
    row,col = key
    matrix_update(m, <int>value, <row_type*><int>row_p[row],
                                 <col_type*><int>col_p[col])
  return <int>m, row_p, col_p

cpdef _construct_matrix(mat):
  cdef matrix* m = matrix_new(0,0)
  return _set_matrix(<int>m, mat)

cpdef _to_scipy_matrix(_m):  
  cdef matrix *m
  cdef vector* col
  cdef vector* row
  cdef matrix_entry* entry
  cdef _ll_item* col_item

  m = <matrix*><int> _m
  cols = to_data_array(<int>m.cols)
  rows = to_data_array(<int>m.rows)

  col_map = dict([(<int>x,i) for i,x in enumerate(cols)])
  row_map = dict([(<int>x,i) for i,x in enumerate(rows)])

  eq_(m.row_count, len(rows))
  eq_(m.col_count, len(cols))
  mat = scipy.sparse.dok_matrix((m.row_count, m.col_count))
  
  for _col in cols:
    col = <vector*><int>_col
    cord = to_data_array(<int>col.store.list)
    for _entry in cord:
      entry = <matrix_entry*><int> _entry
      row = entry.row
      mat[row_map[<int>row], col_map[<int>col]] = entry.value
    
  return mat

cpdef _get_nnz(col_p):
  cdef vector* col
  nnz = 0
  for _col in col_p:
    col = <vector*><int>_col
    cord = to_data_array(<int>col.store.list)
    nnz += len(cord)
  return nnz

import scipy.sparse
import numpy as np

def _assert_matrix_equal(m1, m2):
  eq_(m1.shape, m2.shape)
  r = m1.toarray()-m2.toarray()
  assert (r**2).sum() <= 0.0001, "%r != %r" % (m1.toarray(), m2.toarray())

class TestMatrix:
  def test_create_delete(self):
    m = matrix_new(0,0)
    matrix_delete(m)

  def test_insert_delete_row_col(self):
    m = matrix_new(0,0)
    row_p= [<int>matrix_insert_new_row(m) for i in xrange(3)]
    col_p = [<int>matrix_insert_new_col(m) for i in xrange(2)]
    eq_(len(to_array(<int>m.rows)), 3)
    eq_(len(to_array(<int>m.cols)), 2)
    matrix_remove_col(m, <col_type*><int>col_p[0])
    eq_(len(to_array(<int>m.cols)), 1)
    matrix_remove_row(m, <row_type*><int>row_p[0])
    matrix_remove_row(m, <row_type*><int>row_p[1])
    eq_(len(to_array(<int>m.rows)), 1)
    matrix_delete(m)

  def test_construct(self):
    fixture = \
       [ [0,0,0,0,1],
         [0,1,0,2,0],
         [2,0,1,0,0] ]         
    mat = scipy.sparse.dok_matrix(np.array(fixture, dtype=np.float32))
    m, row_p, col_p = _construct_matrix(mat)
    mat_reverse = _to_scipy_matrix(<int>m)
    _assert_matrix_equal(mat_reverse, mat)
    matrix_delete(<matrix*><int>m)

  def test_update(self):
    fixture = \
       [ [0,0,0,0,1],
         [0,1,0,2,0],
         [2,0,1,0,0] ]         
    mat = scipy.sparse.dok_matrix(np.array(fixture, dtype=np.float32))
    self._m, self.row_p, self.col_p = _construct_matrix(mat)

    cdef matrix *m = <matrix*><int>self._m
    m.squeeze_row = 1
  
    mat[0,4] += 1
    self._update(1, 0, 4)    
    self._assert_equal(mat)

    self._update(1, 0, 2)
    mat[0,2] += 1
    self._assert_equal(mat)

    self._update(-1, 2, 2)
    mat[2,2] += -1
    self._assert_equal(mat)

    mat = scipy.sparse.dok_matrix(mat.toarray()[1:, :])
    self._update(-2, 0, 4)
    self._update(-1, 0, 2)    
    self._assert_equal(mat)
    

  def test_mult_view_create_delete(self):
    cdef matrix *l, *r
    cdef matrix_mult_view *view
    l = matrix_new(0,0)
    r = matrix_new(0,0)
    p = matrix_new(0,0)
    view = mult_view_new(l, r, p)
    mult_view_delete(view)    
    matrix_delete(l)
    matrix_delete(r)
    matrix_delete(p)    


  def test_mult_view_insert_remove(self):
    cdef matrix *l, *r, *p
    cdef matrix_mult_view *view
    l = matrix_new(0,0)
    r = matrix_new(0,0)
    p = matrix_new(0,0)
    view = mult_view_new(l, r, p)
    
    left_shape = (5,4)
    right_shape = (4,10)
    prod_shape = (left_shape[0], right_shape[1])
    lr = [<int>matrix_insert_new_row(l) for i in xrange(left_shape[0])]
    lc = [<int>matrix_insert_new_col(l) for i in xrange(left_shape[1])]
    rc = [<int>matrix_insert_new_col(r) for i in xrange(right_shape[1])]    
    for __m, __s in [(<int>l,left_shape), (<int>r,right_shape),(<int>p, prod_shape)]:
      self._shape(<int>__m, __s)

    matrix_remove_col(l, <vector*><int>lc[3])
    matrix_remove_col(l, <vector*><int>lc[1])    
    
    self._shape(<int>l, (5,2))
    self._shape(<int>r, (2,10))
    self._shape(<int>p, (5,10))

    matrix_remove_row(l, <vector*><int>lr[1])
    matrix_remove_col(r, <vector*><int>rc[3])
    matrix_remove_col(r, <vector*><int>rc[4])
    
    self._shape(<int>l, (4,2))
    self._shape(<int>r, (2,8))
    self._shape(<int>p, (4,8))    

    mult_view_delete(view)    
    matrix_delete(l)
    matrix_delete(r)
    matrix_delete(p)

  def _shape(self, _m, shape):
    cdef matrix *m = <matrix*><int>_m
    eq_((m.row_count, m.col_count), shape)

  def test_mult_view_update(self):
    cdef matrix *l, *r, *p
    cdef matrix_mult_view *view   
    fl = np.array([[1,0], [0,2], [1,3]])
    fr = np.array([[1,0,3], [0,2,0]])
    ml = scipy.sparse.dok_matrix(fl)
    mr = scipy.sparse.dok_matrix(fr)
    l = matrix_new(0,0)
    r = matrix_new(0,0)    
    p = matrix_new(0,0)
    view = mult_view_new(l, r, p)       
    lr = [<int>matrix_insert_new_row(l) for i in xrange(fl.shape[0])]
    lc = [<int>matrix_insert_new_col(l) for i in xrange(fl.shape[1])]
    for key,value in ml.iteritems():
      row,col = key
      matrix_update(l, value, \
          <vector*><int>lr[row], <vector*><int>lc[col])
    
    rr = to_data_array(<int>r.rows)
    eq_(len(rr), fl.shape[1])
    rc = [<int>matrix_insert_new_col(r) for i in xrange(fr.shape[1])]
    for key,value in mr.iteritems():
      row,col = key
      matrix_update(r, value, \
          <vector*><int>rr[row], <vector*><int>rc[col])
    
    self.row_p = to_data_array(<int>p.rows)
    self.col_p = to_data_array(<int>p.cols)
    mp = np.dot(ml,mr)
    self._m = <int>p
    self._assert_equal(mp)
    
    l.squeeze_row = 1
    matrix_update(l, -2, <vector*><int>lr[1], <vector*><int>lc[1])   
    fl = fl[(0,2),:]
    mp = scipy.sparse.dok_matrix(np.dot(fl, fr))
    self._assert_equal(mp)

    r.squeeze_row = 1
    r.squeeze_col = 1
    matrix_update(r, -2, <vector*><int>rr[1], <vector*><int>rc[1])   
    fl = fl[:, 0:1]
    fr = fr[0:1, (0,2)]
    mp = scipy.sparse.dok_matrix(np.dot(fl, fr))
    self._assert_equal(mp)

    mult_view_delete(view)    
    matrix_delete(l)
    matrix_delete(r)
    matrix_delete(p)

  # TODO: A*B*C ... 

  def _assert_equal(self, mat):
    cdef matrix* m = <matrix*><int> self._m
    mat_reverse = _to_scipy_matrix(<int>self._m)
    _assert_matrix_equal(mat_reverse, mat)
    eq_( _get_nnz(to_data_array(<int>m.cols)), mat.getnnz())

  def _update(self, delta, row, col):
    matrix_update(<matrix*><int>self._m, delta,
                  <vector*><int> self.row_p[row],
                  <vector*><int> self.col_p[col])


def assert_mat_equal(_m, mat):
  cdef matrix* m = <matrix*><int> _m
  mat_reverse = _to_scipy_matrix(<int>_m)
  _assert_matrix_equal(mat_reverse, mat)
  eq_( _get_nnz(to_data_array(<int>m.cols)), mat.getnnz())

def _assert_2d_list(list1, list2):
  eq_(len(list1), len(list2))
  for l1,l2 in zip(list1,list2):
    eq_(len(l1),len(l2))
    for a,b in zip(l1,l2):
      eq_(a,b)

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
      model.add_new_document(l, debug = t)
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
    _assert_2d_list(exported, topic_select)

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

