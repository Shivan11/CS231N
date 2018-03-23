import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = np.max(y)+1

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
        image = X[i]
        scores = image.dot(W)
        scores = np.exp(scores)
        total = np.sum(scores)
        prob = scores/total
        loss -= np.log(prob[y[i]])   
        for j in range(num_classes):
            dW[:,j] += image * (prob[j] - (y[i]==j))
        
            
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss/= num_train
  dW/= num_train
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  total_loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores = np.exp(scores)
  total = np.sum(scores,axis=1)
  probs = scores/total[:,None]
  loss = -np.log(probs[range(num_train),y])
  probs[range(num_train),y]-=1
  total_loss = np.sum(loss)/num_train
  total_loss += reg*np.sum(W*W)
  dW = X.T.dot(probs)/num_train
  dW += 2*reg*W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return total_loss, dW

