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

