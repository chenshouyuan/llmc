cdef extern from "lib/hash-table.h":
  struct HashTable:
    pass
  ctypedef void *HashTableKey
  ctypedef void *HashTableValue

  HashTable *hash_table_new()
  void hash_table_free(HashTable *hash_table)
  int hash_table_insert(HashTable *hash_table, 
                        HashTableKey key, 
                        HashTableValue value)
  HashTableValue hash_table_lookup(HashTable *hash_table, 
                                   HashTableKey key)
  int hash_table_remove(HashTable *hash_table, HashTableKey key)
