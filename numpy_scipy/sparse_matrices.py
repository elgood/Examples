#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.sparse import (csr_matrix, csr_matrix, coo_matrix,
                bsr_matrix, dia_matrix, dok_matrix, lil_matrix)
from numpy.random import rand
from time import time
import random

n = 10000 # dimension of matrix


# # List of Lists format (lil_matrix)

# Create a list of list matrix that is of size n x n.  We will fill that one with random values.  Also create lil matrix that is identity.

# In[27]:


lil1 = lil_matrix((n,n))
lilIdentity = lil_matrix((n,n))


# Fill the first one hundred values of the zeroth row with random numbers over the distribution [0, 1).  Then fill the diagonal also with random numbers.

# In[28]:


lil1[0, :100] = rand(100)
lil1.setdiag(rand(n))


# Fill in the identity matrix.

# In[29]:


for i in range(n):
    lilIdentity[i,i] = 1


# See how long it takes to multiply a lil matrix with another lil matrix.

# In[30]:


time1 = time()
lil1 = lil1*lilIdentity
print("Time(s) for lil multiply: " + str(time() - time1))


# Convert them to dense matrices and see how long the multiply takes.

# In[6]:


dense1 = lil1.toarray()
denseIdentity = lilIdentity.toarray()
time1 = time()
dense1 = dense1 * denseIdentity
print("Time(s) for dense multiple: " + str(time() - time1))


# Now how does adding matrices compare?

# In[7]:


lil2 = lil1
time1 = time()
lil2 = lil2 + lil2
print("Time(s) for lil add: " + str(time() - time1))


# In[8]:


dense1 = lil1.toarray()
denseIdentity = lilIdentity.toarray()
time1 = time()
dense1 = dense1 + dense1
print("Time(s) for dense add: " + str(time() - time1))


# What happens when a lil matrix has many values?

# In[9]:


lilManyValues = lil_matrix((n,n))
numValues = int(n*n / 10)
for iter in range(numValues):
    i = random.randrange(n)
    j = random.randrange(n)
    lilManyValues[i,j] = 1
print("Fraction nonzero: " + str(lilManyValues.count_nonzero() / (n*n)))
    
time1 = time()
lilManyValues = lilManyValues * lilManyValues
print("Time for lil multiply: " + str(time() - time1))

denseManyValues = lilManyValues.toarray()
time1 = time()
denseManyValues = denseManyValues * denseManyValues
print("Time for dense multiply: " + str(time() - time1))


# # Compressed Sparse Row (csr_matrix) and Compressed Sparse Column (csc_matrix)

# In[10]:


csr1 = lil1.tocsr()
csc1 = lil1.tocsc()
csrIdentity = lilIdentity.tocsr()
cscIdentity = lilIdentity.tocsc()


# In[11]:


time1 = time()
csr1 = csr1*csr1
print("Time(s) for csr multiply: " + str(time() - time1))


# In[12]:


csr2 = csr1
time1 = time()
csr2 = csr2 + csr2
print("Time(s) for csr add: " + str(time() - time1))


# In[13]:


iters = 10000
time1 = time()
for i in range(iters):
    index = random.randrange(n)
    row = csr1[index,:]
    nnz = row.count_nonzero()
print("Time(s) for accessing rows: " + str(time() - time1))
for i in range(iters):
    index = random.randrange(n)
    row = csr1[:,index]
    nnz = row.count_nonzero()
print("Time(s) for accessing columns: " + str(time() - time1))


# In[14]:


iters = 10000
time1 = time()
for i in range(iters):
    index = random.randrange(n)
    row = csc1[index,:]
    nnz = row.count_nonzero()
print("Time(s) for accessing rows: " + str(time() - time1))
for i in range(iters):
    index = random.randrange(n)
    row = csc1[:,index]
    nnz = row.count_nonzero()
print("Time(s) for accessing columns: " + str(time() - time1))


# # Hindom Optimization Equation 

# F(t+1)=(1/(1+mu)) D^(-1/2) M' D^(-1/2) F(t) + (mu/(1+mu)) Y

# Create something that looks like M' of Hindom.

# In[14]:


n = 200
M = lil_matrix((n,n))
numValues = 500
for iter in range(numValues):
    i = random.randrange(n)
    j = random.randrange(n)
    M[i,j] = random.random() # [0,1)
print("Fraction nonzero: " + str(M.count_nonzero() / (n*n)))
M = M.tocsr()


# Actually have to create an affinity matrix.

# In[15]:


import sys
import math
import matplotlib.pyplot as plt

W = M.toarray()
x_values = []
y_values = []
time1 = time()
for i in range(n):
    for j in range(n):
        if i == j:
            W[i,j] = 0
        else:
            x = M[i,:].toarray() - M[j,:].toarray()
            x = np.linalg.norm(x, ord=2)
            y = math.exp(-pow(x,2))
            #y = math.exp(-x)
            x_values.append(x)
            y_values.append(y)
            W[i,j] = y
print("Time(s) for computing affinity: " + str(time() - time1))
M = W


# Here is a plot of the original normed values with the compute affinity (should look like half a gaussian).

# In[17]:


plt.plot(np.array(x_values), np.array(y_values), 'o', color='black');


# Create diagonal matrix D

# In[18]:


data = np.squeeze(np.asarray(M.sum(axis=1))) # sum the rows
offsets = np.array([0])
D = dia_matrix((data, offsets), shape=(n,n))


# Calculate the exponent (-1/2), which is easy for a diagonal.  We can just take each diagonal element and raise it to (-1/2).

# In[19]:


D = D.tocsr() # Convert to csr because can't use subscripting
for i in range(n):
    D[i,i] = D[i,i] ** (-1/2) 


# Create the matrix S

# In[20]:


S = D * M * D


# mu is just how we weight each piece, smoothness vs fidelity to known labels

# In[21]:


mu = 0.95
alpha = 1/(1 + mu)
beta = mu/(1 + mu)


# Create some labels. 

# In[22]:


Y = np.zeros((n,2))
# Set some to be malicious and benign
for i in range(int(n/10)):
    index = random.randrange(n)
    if i % 2 == 0:
        Y[index,0] = 1
        Y[index,1] = 0
    else:
        Y[index,0] = 0
        Y[index,1] = 1


# Set F to be Y.

# In[23]:


F = Y
print(F)


# Do one iteration

# In[24]:


F = alpha * S.dot(F) + beta * Y
print(F)


# In[25]:


for i in range(10000):
  F = alpha * S.dot(F) + beta * Y
  
print(F)


# In[ ]:




