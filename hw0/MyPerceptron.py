
# Header
import numpy as np


# The major Perceptron algorithm
# Input:
#   X: all data sample features, with the shape nx2
#   y: ground truth class labels, shape n
#   w0: initla weight, set as [1,-1]
# Return:
#   w: weight at convergence, shape 2
#   initial_error_rate: error rate before updating w, 1
#   predict_error_rate: error rate after convergence, 1
def MyPerceptron(X,y,w0):
    w=w0
    converge = False

    # compute initial error rate
    initial_error_rate = computeError(y, predict(w, X))

    while not converge:
        flag = False  # flag denotes whether w is updated at least once
        for t in range(len(X)):
            prev_w = w
            # update w if necessary
            w = updateW(w, X[t], y[t])
            # check whether w is updated at this round
            if not flag and np.linalg.norm(w-prev_w)>1e-10:
                flag = True
        converge = not flag

    # compute final error rate after convergence
    predict_error_rate = computeError(y, predict(w, X))

    return (w, initial_error_rate, predict_error_rate)

# Input:
#   w: current weight, with shape 2
#   xt: the features of t-th sample, 2
#   yt: the ground-truth label of t-th sample, 1
# Return:
#   new_w: w after update (if need), 2*1
#   update_flag: a boolean flag denote whether the w is updated, 1
def updateW(w, xt, yt):
    # Implement the line 4-5 from the algorithm to update the w base on xt and yt
    new_w = w   # placehold to store the updated w
    # Implement the If statement here based on Line 4-5 of the pseudo-code
    if yt * np.dot(w,xt) <= 0:
        new_w = new_w + yt*xt

    return new_w


# Input:
#   w: current weight, with shape 2
#   X: all data sample features, with the shape nx2
# Return:
#   y_hat: predicted y labels, each element will be +1 or -1, y_hat has shape n
def predict(w, X):
    y_hat = 0  # placehold to store predict result y_hat, can ignore this line
    # compute the inner product of w and X, and convert the value into a list of -1/+1 based on the sign
    w_inner_X = np.inner(w,X)

    for i in range(len(w_inner_X)):
        if w_inner_X[i] >= 0:
            w_inner_X[i] = 1
        else:
            w_inner_X[i] = -1

    y_hat = w_inner_X
    return y_hat


# Input:
#   y: ground truth y labels, each element will be +1 or -1, y has shape n
#   y_hat: predicted y labels, each element will be +1 or -1, y_hat has shape n
# Return:
#   e: float number error rate, has shape 1
def computeError(y, y_hat):
    e = 0.0  # placehold to store the error rate w, can ignore this line
    # compute the error rate: y is the groundtruth and y_hat is the predictions
    # e.g if y=[1,1,1,1,-1], y_hat = [1,1,1,1,1], there is 1 error, the error rate is 1/5 = 0.20
    incorrect = 0

    for i in range(len(y)):
        if y[i] != y_hat[i]:
            incorrect += 1

    e = incorrect/len(y)
    return e
