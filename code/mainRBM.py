import numpy as np
import rbm
import projectLib as lib
import csv

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5
'''
# SET PARAMETERS HERE!!!
F = 8
alpha = 0.03
momentum = 0
batch_size = 10
reg = 1e-4
'''

epochs = 20
#gradientLearningRate = 0.001

# Parameter tuning
m_range = [0.6, 0.75, 0.9]
r_range = [3e-4, 1e-3, 1e-2]
a_range = [1e-1]
b_range = [5, 10]
f_range = [20]

total = len(m_range) * len(r_range) * len(a_range) * len(b_range) * len(f_range)

def getBatches(array, batch_size):
    ret = []
    for i in range(int(len(array)/batch_size)):
        ret.append(array[i*batch_size:i*batch_size+batch_size])
    if len(array)%batch_size != 0:
        ret.append(array[len(array)/batch_size:])
    return ret

# output file
csvfile = open("fine_tuning.csv",'w')
writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['Momentum', 'Regularization', 'Alpha', 'Batch Size', 'F', 'Epoch', 'RMSE'])

best_momentum = 0
best_reg = 0
best_epoch = 0
best_alpha = 0
best_batchsize = 0
best_F = 0
counter = 0

for momentum in m_range:
    for reg in r_range:
        for alpha in a_range:
            for B in b_range:
                for F in f_range:
                    # reset best params
                    min_rmse = 2
                    print("Finished: %.3f" % (1.0 * counter / total))
                    counter += 1
                    
                    # Initialise all our arrays
                    W = rbm.getInitialWeights(trStats["n_movies"], F, K)
                    grad = np.zeros(W.shape)
                    posprods = np.zeros(W.shape)
                    negprods = np.zeros(W.shape)

                    for epoch in range(1, epochs+1):
                        # in each epoch, we'll visit all users in a random order
                        visitingOrder = np.array(trStats["u_users"])
                        np.random.shuffle(visitingOrder)
                        
                        ###########################################################################
                        #                          EXTENSION: ADAPTIVE LEARNING RATE              #
                        ###########################################################################
                        #adaptiveLearningRate = alpha / epoch
                        adaptiveLearningRate = alpha * np.exp(-alpha * epoch)
                        
                        ###########################################################################
                        #                             EXTENSION: MINI BATCH                       #
                        ###########################################################################
                        batches = getBatches(visitingOrder, B)
                        for batch in batches:
                            prev_grad = grad
                            grad = np.zeros(W.shape)
                            for user in batch:
                                # get the ratings of that user
                                ratingsForUser = lib.getRatingsForUser(user, training)

                                # build the visible input
                                v = rbm.getV(ratingsForUser)
                                
                                # get the weights associated to movies the user has seen
                                weightsForUser = W[ratingsForUser[:, 0], :, :]
                        
                                ### LEARNING ###
                                # propagate visible input to hidden units
                                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
                                # get positive gradient
                                # note that we only update the movies that this user has seen!
                                posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)
                        
                                ### UNLEARNING ###
                                # sample from hidden distribution
                                sampledHidden = rbm.sample(posHiddenProb)
                                # propagate back to get "negative data"
                                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
                                # propagate negative data to hidden units
                                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
                                # get negative gradient
                                # note that we only update the movies that this user has seen!
                                negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)
                                
                                ###########################################################################
                                #                             EXTENSION: REGULARIZATION                   #
                                ###########################################################################
                                grad += adaptiveLearningRate * ((posprods - negprods) / trStats["n_users"] - reg * W)
                        
                            ###########################################################################
                            #                             EXTENSION: MOMENTUM                         #
                            ###########################################################################
                            W += grad + momentum * prev_grad
                            
                        # Print the current RMSE for training and validation sets
                        # this allows you to control for overfitting e.g
                        # We predict over the training set
                        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
                        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
                    
                        # We predict over the validation set
                        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
                        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
                        
                        print("### EPOCH %d ###" % epoch)
                        print("### momentum %.1f regularization %.5f alpha %.3f B %d F %d epoch %d ###" % \
                              (momentum, reg, alpha, B, F, epoch))
                        print("Training loss = %f" % trRMSE)
                        print("Validation loss = %f" % vlRMSE)
                        
                        ###########################################################################
                        #                             EXTENSION: EARLY STOPPING                   #
                        ###########################################################################
                        if vlRMSE < min_rmse:
                            best_momentum = momentum
                            best_reg = reg
                            best_epoch = epoch
                            best_alpha = alpha
                            best_B = B
                            best_F = F
                            min_rmse = vlRMSE
                    print('Best RMSE: ', min_rmse)
                    writer.writerow([best_momentum, best_reg, best_alpha, best_B, best_F, best_epoch, min_rmse])
                    
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
np.savetxt("predictedRatings.txt", predictedRatings)
                        
                        
'''
    for user in visitingOrder:
        # get the ratings of that user
        ratingsForUser = lib.getRatingsForUser(user, training)

        # build the visible input
        v = rbm.getV(ratingsForUser)

        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :]

        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        grad[ratingsForUser[:, 0], :, :] = gradientLearningRate * (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])
        W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :]

    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    print("### EPOCH %d ###" % epoch)
    print("Training loss = %f" % trRMSE)
    print("Validation loss = %f" % vlRMSE)
'''
