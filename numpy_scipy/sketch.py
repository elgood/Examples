import numpy as np
import argparse
import sklearn.preprocessing.normalize

""" This example shows how to perform sketches, which translates
 a vector of size n to a vector of size k (k << n) by performing
 the dot product of the vector with the k sketch vectors, resulting
 in a smaller vector of length k, but which preserves the L2 norm.

 Description of this process can be found on page 184 of "Data Streams:
 Models and Algorithms", Edited by Charu C. Aggarwal, Springer.
"""

def get_sketch_vectors(n: int, k: int) -> np.ndarray:
  """ Gets k sketch vectors of length n.  

  Parameters:
  n: int - The length of the sketch vectors.
  k: int - The number of sketch vectors. 
  """

  # We will store the k sketch vector in a kxn array.
  # Array elements are drawn from a normal distribution with 
  # zero mea and unit variance.
  sketch_vectors = np.random.normal(size=(k,n))

  # Each sketch vector is normalized to a magnitude of one
  for i in range(k):
    normalize(sketch_vectors[i,:], copy=False



def main():
  
  parser = argparse.ArgumentParser("Produces sketch vectors and " +
    "applies them a vector and compares the norm of the original " +
    "vector with the transformed vector.")
  parser.add_argument("n", type=int, help="The length of the " +
    "original vector.")
  parser.add_argument("k", type=int, help="The length of the " +
    "transformed vector.")
  
  FLAGS = parser.parse_args()


  
  


if __name__ == '__main__'
  main()
