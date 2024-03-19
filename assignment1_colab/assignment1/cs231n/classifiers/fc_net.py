from builtins import range
from builtins import object
import numpy as np

# from ..layers import *
# from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        np.random.seed(0)

        self.params['W1'] = np.random.normal(0, weight_scale, size= (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(shape= (1, hidden_dim))

        self.params['W2'] = np.random.normal(0, weight_scale, size= (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(shape= (1, num_classes))

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        z = X.dot(self.params['W1']) + self.params['b1']
        z = np.maximum(0, z)
        scores = z.dot(self.params['W2']) + self.params['b2']

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # from softmax import softmax_loss_vectorized
        scores -= scores.max(axis=-1)[:, None]
        # print(scores)

        scores = np.exp(scores) / np.sum(np.exp(scores), axis= -1)[:, None]
        scores_y = scores[range(len(scores)), y] # N,
        loss = -np.log(scores_y)
        loss = np.sum(loss) / len(X) + 0.5 * self.reg * (np.sum(W1* W1) + np.sum(W2* W2))

        scores[range(len(scores)), y] -= 1
        scores /= len(X)
        dW2 = z.T.dot(scores) + self.reg*W2
        db2 = np.sum(scores, axis= 0)

        mask = np.ones_like(z)
        mask[z == 0] = 0
        dW1 = X.T.dot((scores).dot(W2.T) * mask) + self.reg*W1
        db1 = np.sum(scores.dot(W2.T) * mask, axis= 0)

        grads.update(W1= dW1, b1= db1, W2= dW2, b2= db2)
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

if __name__ == '__main__':
    model = TwoLayerNet()
    from neural_net import TwoLayerNet
    model2 = TwoLayerNet()
    

    N = 5
    D = 32**2 * 3
    num_classes = 10

    X = np.random.rand(N, D)
    y = np.random.randint(num_classes, size= (N, ))

    loss, grads = model.loss(X, y)
    print(grads['b2'][:10])
    loss, grads = model2.loss(X, y)
    print(grads['b2'][:10])
