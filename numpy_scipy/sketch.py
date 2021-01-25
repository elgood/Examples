import numpy as np
import argparse
import math

""" This example shows how to perform sketches, which translates
 a vector of size n to a vector of size k (k << n) by performing
 the dot product of the vector with the k sketch vectors, resulting
 in a smaller vector of length k, but which preserves the L2 norm.

 Description of this process can be found on page 184 of "Data Streams:
 Models and Algorithms", Edited by Charu C. Aggarwal, Springer.

 Original reference: "Identifying Representative Trends in Massive Time
 Series Data Sets Using Sketches" by Indyk, Koudas, and Muthukrishnan.
"""

def get_sketch_vectors(n: int, k: int) -> np.ndarray:
  """ Gets k sketch vectors of length n.  Returned numpy array has
    the sketch vectors as rows.

  Parameters:
  n: int - The length of the sketch vectors.
  k: int - The number of sketch vectors. 

  Return:
  sketch_vectors: np.ndarray - The sketch vectors as a numpy array.
  """

  # We will store the k sketch vector in a kxn array.
  # Array elements are drawn from a normal distribution with 
  # zero mea and unit variance.
  sketch_vectors = np.random.normal(size=(k,n))

  # Each sketch vector is normalized to a magnitude of one
  for i in range(k):

    # Change in place
    sketch_vectors[i,:] = (sketch_vectors[i,:] / 
                            np.linalg.norm(sketch_vectors[i,:]))

  return sketch_vectors

def apply_sketch_vectors(vectors: np.ndarray, sv: np.ndarray) -> np.ndarray:
  """ Applies the sketch vectors to input vectors to transform
    them from length n to length k.
  """

  num = vectors.shape[0]
  k = sv.shape[0]
  rv = np.zeros((num, k))

  # Probably a better way to write this
  for i in range(num):
    for j in range(k):
      rv[i, j] = vectors[i,:].dot(sv[j,:])

  return rv


def main():
  
  parser = argparse.ArgumentParser("Produces sketch vectors and " +
    "applies them a vector and compares the norm of the original " +
    "vector with the transformed vector.")
  parser.add_argument("n", type=int, help="The length of the " +
    "original vectors.")
  parser.add_argument("k", type=int, help="The length of the " +
    "transformed vector.")
  parser.add_argument("num", type=int, help="How many of the original " +
    "vectors to make.")  
  
  FLAGS = parser.parse_args()

  n = FLAGS.n
  k = FLAGS.k
  num = FLAGS.num

  sketch = get_sketch_vectors(FLAGS.n, FLAGS.k)
  
  vectors = np.random.random((num, n))
  
  new_vectors = apply_sketch_vectors(vectors, sketch) 

  distances_orig = []
  distances_new  = []

  eps = math.sqrt( 9 * math.log(n) / k) 

  for i in range(num):
    for j in range(num):
      if i != j:
        d1 = np.linalg.norm(vectors[i,:] - vectors[j,:])
        d2 = np.linalg.norm(new_vectors[i,:] - new_vectors[j,:])
        low = d1 * (1-eps)
        up  = d1 * (1+eps)
        print("% 2d,% 2d orig % 5.2f new % 5.2f, expected(% 5.2f, % 5.2f)" 
             %(i, j, d1, d2, low, up))
              
        distances_orig.append(d1)
        distances_new.append(d2)

  factor = math.sqrt(9*math.log(n) / k)
  print("Distances can be off by a factor of " + str(factor))
        
        

if __name__ == '__main__':
  main()
