from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(len(X)):
        # X[i] D,
        scores = X[i].dot(W)
        scores -= scores.max()
        scores = np.exp(scores)/np.sum(np.exp(scores)) # 1, 3
        loss += -np.log(scores[y[i]])

        dW[:, y[i]] += (scores[y[i]] - 1) * X[i]

        for j in range(W.shape[1]):
            if j == y[i]:
                continue
            dW[:, j] += scores[j] * X[i]

    loss /= len(X)
    loss += reg * np.sum(W * W)
    dW /= len(X)
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    scores -= np.max(scores, axis= -1)[:, None]

    scores = np.exp(scores) / np.sum(np.exp(scores), axis= -1)[:, None] # N C
    loss = -y * np.log(scores[range(len(scores)), y])
    loss = np.sum(loss) / len(X) + reg * np.sum(W * W)

    scores[range(len(scores)), y] -= 1
    dW = X.T.dot(scores) / len(X) + 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


if __name__ == "__main__":
    np.random.seed(0)
    N = 3
    C = 3
    D = 5
    reg = 1
    X = np.random.rand(N, D)
    W = np.random.rand(D, C)
    y = np.random.randint(C, size= (N))

    # scores = X.dot(W)
    # print(scores)
    # print(y)
    # print(scores[range(len(X)), y])

    loss, grad = softmax_loss_naive(W, X, y, reg)
    print(grad)
    loss, grad = softmax_loss_vectorized(W, X, y, reg)
    print(grad)