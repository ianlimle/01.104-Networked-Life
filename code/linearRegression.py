import numpy as np
import projectLib as lib
import copy

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    ###########################################################################
    #                             START OF YOUR CODE                          #
    ###########################################################################
    for i in range(trStats["n_ratings"]):
        A[i][training[i][0]] = 1
        A[i][training[i][1]] = 1
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return A

# we also get c
def getc(rBar, ratings):
    ###########################################################################
    #                             START OF YOUR CODE                          #
    ###########################################################################
    rate = copy.deepcopy(ratings)
    for i in range(len(ratings)):      
        rate[i] = rate[i] - rBar
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    ###########################################################################
    #                             START OF YOUR CODE                          #
    ###########################################################################
    I = np.identity(trStats["n_movies"] + trStats["n_users"],dtype=None)
    AT = np.transpose(A)
    ATA = np.dot(AT,A)
    lI = 0.000000000001*I
    inverse = np.linalg.inv(ATA+lI) 
    ATc = np.matmul(AT,c)
    b = np.matmul(inverse,ATc)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    ###########################################################################
    #                             START OF YOUR CODE                          #
    ###########################################################################
    I = np.identity(trStats["n_movies"] + trStats["n_users"],dtype=None)
    AT = np.transpose(A)
    ATA = np.matmul(AT,A)
    lI = l*I
    inverse = np.linalg.inv(ATA+lI)    
    ATc = np.matmul(AT,c)
    b = np.matmul(inverse,ATc)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)

# Regularised version
l = 1
b = param_reg(A, c, l)

print("Linear regression, l = %f" % l)
print("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))
