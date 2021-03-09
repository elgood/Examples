# You can have import statements like this if your PYTHONPATH environment
# variable is set appropriately.  I usually set it at the end of the
# activate script (e.g. <virtualenv>/bin/activate).  That way when I 
# activate the virtual environment, the PYTHONPATH is set properly.
from numpy_scipy import sketch

import unittest
import numpy as np
import math

class TestSketch(unittest.TestCase):

  # Add code here that will run before every test case
  def setUp(self):
    pass

  # Add code here that will run after every test case
  def tearDown(self):
    pass

  def test_sketch(self):
    n = 5
    k = 2
    vectors = sketch.get_sketch_vectors(n, k) 
    self.assertEqual(vectors.shape, (k, n))

    for v in vectors:
      self.assertTrue(abs(np.linalg.norm(v) - 1.0) < 0.001)



# Won't run tests unless we have this:
if __name__ == '__main__':
  unittest.main()




