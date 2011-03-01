from llmc.py_spmatrix import Matrix, ProdMatrix
from scipy.sparse import dok_matrix, rand as rand_matrix
from nose.tools import *
import numpy as np
from numpy.testing import assert_array_equal

def create_from_scipy(row=10, col=8):
  dok = rand_matrix(row, col, density=0.2, format="dok")
  array = dok.toarray()
  matrix = Matrix(matrix=dok)
  return array, dok, matrix

def check_attribs(array, dok, matrix):
  eq_(matrix.shape[0], dok.shape[0])
  eq_(matrix.shape[1], dok.shape[1])

def eq_mat(matrix, dok_matrix):
  eq_(matrix.shape, dok_matrix.shape)
  assert_array_equal(matrix.to_matrix().toarray(), dok_matrix.toarray())

def test_entries():
  array, dok, matrix = create_from_scipy()
  check_attribs(array, dok, matrix)

  rows = matrix.getrows()
  cols = matrix.getcols()

  #  .. testing __getitems__ ..
  for position, value in dok.iteritems():
    row, col = position
    eq_(matrix[rows[row],cols[col]], value)
  eq_mat(matrix, dok)

  # .. testing __setitems__ ..
  plus = rand_matrix(dok.shape[0], dok.shape[1], density=0.1, format="dok")
  dok = dok+plus
  for position, value in plus.iteritems():
    row, col = position
    matrix[rows[row], cols[col]] += value
  eq_mat(matrix, dok)

  # .. testing update ..
  plus = rand_matrix(dok.shape[0], dok.shape[1], density=0.1, format="dok")
  dok = dok+plus
  for position, value in plus.iteritems():
    row, col = position
    matrix.update(rows[row],cols[col],value)
  eq_mat(matrix, dok)

def test_append_remove():
  array, dok, matrix = create_from_scipy()
  rows_to_remove = [2,4,7]
  cols_to_remove = [1,3,5]

  rows = matrix.getrows()
  cols = matrix.getcols()

  oarray = array
  for row in rows_to_remove:
    matrix.remove_row(rows[row])
    rows.pop(row)
    array = np.delete(array, row, 0)
    dok = dok_matrix(array)
    eq_mat(matrix, dok)
    check_iters(matrix, dok)

  for col in cols_to_remove:
    matrix.remove_col(cols[col])
    cols.pop(col)
    array = np.delete(array, col, 1)
    dok = dok_matrix(array)
    eq_mat(matrix, dok)
    check_iters(matrix, dok)

  n_append_row = 5
  n_append_col = 5

  cols = matrix.getcols()
  for row in xrange(n_append_row):
    new_rand = rand_matrix(1, array.shape[1], 0.2, format="dok")
    print list(new_rand.iteritems())
    new = dict([(cols[k[1]], v) for k, v in new_rand.iteritems()])
    matrix.append_row(new)
    array = np.insert(array, array.shape[0], new_rand.toarray(), 0)
    dok = dok_matrix(array)
    eq_mat(matrix, dok)
    check_iters(matrix, dok)

  rows = matrix.getrows()
  for col in xrange(n_append_col):
    new_rand = rand_matrix(array.shape[0], 1, 0.2, format="dok")
    new = dict([(rows[k[0]], v) for k, v in new_rand.iteritems()])
    matrix.append_col(new)
    array = np.insert(array, array.shape[1], new_rand.T.toarray(), 1)
    dok = dok_matrix(array)
    eq_mat(matrix, dok)
    check_iters(matrix, dok)

def test_iters():
  array, dok, matrix = create_from_scipy()
  check_iters(matrix, dok)

def check_iters(matrix, dok):
  #assert_array_equal(matrix.listrows(), xrange(dok.shape[0]))

  rows = matrix.getrows()
  rowmap = dict([(x,i) for i,x in enumerate(rows)])
  cols = matrix.getcols()
  colmap = dict([(x,i) for i,x in enumerate(cols)])

  # .. check through rows ..
  for row in matrix.listrows():
    col_dok = dok_matrix((1, matrix.shape[1]))
    for col, value in matrix.listcols(row):
      col_dok[0, colmap[col]] = value
    assert_array_equal(col_dok.toarray(), dok[rowmap[row],:].toarray())

  #assert_array_equal(matrix.listcols(), xrange(dok.shape[1]))
  # .. check through cols ..
  for col in matrix.listcols():
    row_dok = dok_matrix((matrix.shape[0], 1))
    for row, value in matrix.listrows(col):
      row_dok[rowmap[row], 0] = value
    assert_array_equal(row_dok, dok[:, colmap[col]])


def test_dot():
  dl = rand_matrix(10, 8, density=0.2, format="dok")
  dr = rand_matrix(8, 9, density=0.2, format="dok")
  dp = np.dot(dl, dr)

  # .. create prod matrix ...
  ml, mr = Matrix(), Matrix()
  mp = ProdMatrix(ml, mr)

  # .. setting values ..
  ml.from_matrix(dl)
  mr.set_cols(dr.shape[1])
  mr.set_matrix(dr)
  eq_mat(mp, dp)

  # .. adding right rows/left cols ...

