from sklearn import datasets, model_selection, metrics

import matplotlib.pyplot as plt
import numpy as np
import random
import time

def fit_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"Time taken to train the model: {round(time.time()-start, 6)} seconds")
    return wrapper

class LinearRegression:
    '''
    This class is used to represent a linear regression module.

    Attributes:
        max_iter (int): the maximum iterations to train the model.
        w (array): the weights of the model.
        b (float): the bias of the model.
        mean_X [array]: the mean values of each feature to be used in standardizing data.
        std_X [array]: the standard deviation values of each feature to be used in standardizing data.
    '''
    def __init__(self, max_iterations=3000):
        self.max_iter = max_iterations
        self.w = []
        self.b = None
        self.mean_X = []
        self.std_X = []

    def _initialise(self, n_features):
        """ Initialises the random weights and bias of the model with the correct dimensions

        Args:
            n_features (int): the number of features in the data.
        """
        self.w = np.random.randn(n_features)
        self.b = np.random.rand()

    def _standarize(self, X):
        """ Standardizes the data. 

        Args:
            X (array): the data to be standardized.

        Returns:
            standardized_X: the standardized data
        """
        if len(self.mean_X) == 0:
            self.mean_X = np.mean(X, axis=0)
            self.std_X = np.std(X, axis=0)

        standardized_X = (X - self.mean_X) / self.std_X
        return standardized_X

    def _calculate_loss(self, X, y):
        """ Calculates the Mean squared error loss.

        Args:
            X (array): the matrix of feature data.
            y (array): the array of ground truth.

        Returns:
            average_loss: the average MSE loss.
        """
        m = len(y)
        y_hat = X @ self.w + self.b
        diff = y_hat - y
        average_loss = np.dot(diff, diff) / m

        return average_loss

    def _step(self, X, y, lr):
        """ Updates the parameters of the model using gradient descent.

        Args:
            X (array): the feature data.
            y (array): the label data.
            lr (float): the learning rate.
        """
        y_hat = X @ self.w + self.b
        diffs = y_hat - y
        m = len(y)

        dLdw = X.T @ diffs * 2 / m
        dLdb = np.sum(diffs) * 2 / m

        self.w = self.w - lr*dLdw
        self.b = self.b - lr*dLdb

    def _get_minibatch(self, X, y, size):
        """ Returns a minibatch of the data to be used in minibatch stochastic gradient descent. 

        Args:
            X (array): the feature data.
            y (array): the label data.
            size (int): the size of the minibatch

        Returns:
            new_X, new_y: the minibatch data.
        """
        total = len(y)
        used_indices = set()
        while len(used_indices) < size:
            ind = random.randrange(0, total)
            try:
                try:
                    if ind not in used_indices:
                        new_X = np.vstack([new_X, X[ind,:]])
                        new_y = np.vstack([new_y, y[ind]])
                        used_indices.add(ind)
                except:
                    new_X = X[ind,:]
                    new_y = y[ind]
                    used_indices.add(ind)
            except:
                try:
                    if ind not in used_indices:
                        new_X = np.vstack([new_X, X[ind]])
                        new_y = np.vstack([new_y, y[ind]])
                        used_indices.add(ind)
                except:
                    new_X = X[ind]
                    new_y = y[ind]
                    used_indices.add(ind)

        new_y = new_y.reshape(-1)
        return new_X, new_y

    @fit_timer
    def fit(self, X, y, lr=0.001, minibatch_size='all_data', verbose=True):
        """ Fits the model parameters to the input data. 

        Args:
            X (array): the feature data.
            y (array): the label data.
            lr (float): the learning rate.
            minibatch_size (int): the size of the batches to be used in minibatch gradient descent.
            verbose (bool): indicates whether an update on the loss should be printed every 100 iterations.
        """
        n_features = X.shape[1]
        m = len(y)
        self._initialise(n_features)

        X = self._standarize(X)
        all_losses = []

        for i in range(self.max_iter):

            if type(minibatch_size) != str and minibatch_size < m:
                X, y = self._get_minibatch(X, y, minibatch_size)

            loss = self._calculate_loss(X, y)
            all_losses.append(loss)

            if verbose == True:
                if i % 100 == 0:
                    print(f'Iteration {i}, Loss = {loss}')

            self._step(X, y, lr)

            if len(all_losses) > 1:
                if (all_losses[-2] - all_losses[-1])/all_losses[-2] < 0.00001:
                    break

        self._losses = all_losses

    def score(self, X, y):
        """ Caclulates the R2 score for some input feature data. 

        Args:
            X (array): the feature data.
            y (array): the label data.

        Returns:
            r2 score: A score of how good the model is at fitting the data -> [-inf,1]
        """
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        y_hat = X @ self.w + self.b

        score = metrics.r2_score(y, y_hat)

        return score

    def predict(self, X):
        """ Returns the predicted labels for input feature data 

        Args:
            X (array): the feature data.

        Returns:
            y_hat: the predicted output.
        """
        if len(self.w) == 0:
            raise Exception('Need to fit the model to data.')

        if len(self.w) != X.shape[1]:
            raise Exception(f'Input data should be of shape (n_sample, {len(self.w)}).')

        X = self._standarize(X)

        y_hat = X @ self.w + self.b

        return y_hat
        
if __name__ == "__main__":

    X, y = datasets.fetch_california_housing(return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    losses = model.fit(X_train, y_train)
    
    print(model.score(X_test, y_test))

    plt.figure()
    plt.plot(model._losses)
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title('Binary Cross Entropy loss with logits vs Iterations')
    plt.show()