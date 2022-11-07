import numpy as np

class GaussianDiscriminant_C1:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))   # S1 and S2, store in 2*(8*8) matrices
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)

        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        predictions = np.zeros(Xtest.shape[0])
        g1 = np.zeros(Xtest.shape[0])
        g2 = np.zeros(Xtest.shape[0])

        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finally get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        for i in range(Xtest.shape[0]):
            g1[i] = (-1/2)*np.log(abs(np.linalg.det(self.S[0,:,:]))) - (1/2)*((Xtest[i,:] - self.m[0,:]).T.dot((np.linalg.inv(self.S[0,:,:]).dot(Xtest[i,:] - self.m[0,:]))) + np.log(self.p[0]))
            g2[i] = (-1/2)*np.log(abs(np.linalg.det(self.S[1,:,:]))) - (1/2)*((Xtest[i,:] - self.m[1,:]).T.dot((np.linalg.inv(self.S[1,:,:]).dot(Xtest[i,:] - self.m[1,:]))) + np.log(self.p[1]))

        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        for i in range(g1.shape[0]):
            if g1[i] > g2[i]:
                predictions[i] = 1
            else:
                predictions[i] = 2

        return predictions


class GaussianDiscriminant_C2:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))  # S1 and S2, store in 2*(8*8) matrices
        self.shared_S =np.zeros((d,d))  # the shared covariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Step 3: Compute the shared covariance matrix that is used for both class
        self.shared_S = self.p[0]*self.S[0,:,:]+self.p[1]*self.S[1,:,:]

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        predictions = np.zeros(Xtest.shape[0])
        g1 = np.zeros(Xtest.shape[0])
        g2 = np.zeros(Xtest.shape[0])

        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finally get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)

        for i in range(Xtest.shape[0]):
            g1[i] = (-1/2)*((Xtest[i,:] - self.m[0,:]).T.dot((np.linalg.inv(self.shared_S)))).dot((Xtest[i,:] - self.m[0,:])) + np.log(self.p[0])
            g2[i] = (-1/2)*((Xtest[i,:] - self.m[1,:]).T.dot((np.linalg.inv(self.shared_S)))).dot((Xtest[i,:] - self.m[1,:])) + np.log(self.p[1])

        # Fill in your code here !!!!!!!!!!!!!!!!!!!!!!!
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        for i in range(g1.shape[0]):
            if g1[i] > g2[i]:
                predictions[i] = 1
            else:
                predictions[i] = 2

        return predictions


class GaussianDiscriminant_C3:
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))  # S1 and S2, store in 2*(8*8) matrices
        self.shared_S =np.zeros((d,d))  # the shared covariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        self.S[0,:,:] = computeCov(Xtrain1)
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        self.S[1,:,:] = computeCov(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # Step 3: Compute the diagonal of S1 and S2, since we assume each feature is independent, and has non diagonal entries cast to 0
        # [[1,2],[2,4]] => [[1,0],[0,4]], try np.diag() twice
        self.S[0,:,:] = np.diag(np.diag(self.S[0,:,:]))
        self.S[1, :, :] = np.diag(np.diag(self.S[1, :, :]))

        # Step 4: Compute the shared covariance matrix that is used for both class
        self.shared_S = np.add(self.p[0]*self.S[0,:,:], self.p[1]*self.S[1,:,:])
        self.s = np.var(Xtrain, axis = 0)

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])
        g1 = np.zeros(Xtest.shape[0])
        g2 = np.zeros(Xtest.shape[0])

        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finally get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        # Please note here, currently we assume shared_S is a d*d diagonal matrix, the non-capital si^2 in the lecture formula will be the i-th entry on the diagonal

        for i in range(Xtest.shape[0]):  # for each test set example
            g1[i] = -1/2 * np.sum(((Xtest[i,:] - self.m[0,:]) / np.sqrt(self.s)) ** 2) + np.log(self.p[1])
            g2[i] = -1/2 * np.sum(((Xtest[i,:] - self.m[1,:]) / np.sqrt(self.s)) ** 2) + np.log(self.p[1])

            print(i, g1[i],g2[i])

        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]
        for i in range(g1.shape[0]):
            if g1[i] > g2[i]:
                predictions[i] = 1
            else:
                predictions[i] = 2

        return predictions


# ------------------------------------- Helper Functions start from here --------------------------------------------------------------
# Input:
# features: n*d matrix (n is the number of samples, d is the number of dimensions of the feature)
# labels: n vector
# Output:
# features1: n1*d
# features2: n2*d
# n1+n2 = n, n1 is the number of class1, n2 is the number of samples from class 2
def splitData(features, labels):
    # placeholders to store the separated features (feature1, feature2),
    features1 = np.zeros([np.sum(labels == 1),features.shape[1]])  
    features2 = np.zeros([np.sum(labels == 2),features.shape[1]])

    # separate the features according to the corresponding labels, for example
    # if features = [[1,1],[2,2],[3,3],[4,4]] and labels = [1,1,1,2], the resulting feature1 and feature2 will be
    # feature1 = [[1,1],[2,2],[3,3]], feature2 = [[4,4]]

    featuresize1 = 0
    featuresize2 = 0

    for i in range(features.shape[0]):

        if labels[i] == 1:
            features1[featuresize1,:] = features[i,:]
            featuresize1 += 1

        else:
            features2[featuresize2,:] = features[i,:]
            featuresize2 += 1


    return features1, features2


# compute the mean of input features
# input: 
# features: n*d
# output: d
def computeMean(features):
    # placeholders to store the mean for one class
    m = np.zeros(features.shape[1])

    # try to explore np.mean() for convenience
    m = np.mean(features,axis=0)
    return m


# compute the mean of input features
# input: 
# features: n*d
# output: d*d
def computeCov(features):
    # placeholders to store the covariance matrix for one class
    S = np.eye(features.shape[1])

    S = np.cov(features.T)
    return S


# compute the priors of input features
# input: 
# features: n
# output: 2
def computePrior(labels):
    # placeholders to store the priors for both class
    p = np.array([0.5,0.5])

    # p1 = numOf class1 / numOf all the data; same as p2

    p1 = 0
    p2 = 0
    last = ""

    for i in range(labels.shape[0]):
        if labels[i] == 1:
            p1 += 1
        else:
            p2 += 1

    p[0] = p1/(p1+p2)
    p[1] = p2/(p1+p2)
    return p
