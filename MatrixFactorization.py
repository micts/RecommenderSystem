import numpy as np

# load data
ratings = np.genfromtxt("ml-1m/ratings.dat",usecols = (0,1,2),delimiter = "::",dtype = 'int')

# split data into 5 train and test folds
nfolds = 5

# allocate memory for results:
err_train = np.zeros(nfolds)
err_test = np.zeros(nfolds)

# to make sure you are able to repeat results, set the random seed to something:
np.random.seed(17)

# create a sequence of indices for each of the 5 folds 
seqs = [x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

# for each fold:
for fold in range(nfolds):
    train_index = np.array([x != fold for x in seqs])
    test_index = np.array([x == fold for x in seqs])
    train = ratings[train_index] 
    test = ratings[test_index]
    
    #x_hat = train[:,0:2]
    #x_hat = np.hstack((x_hat,np.random.randint(1,6,size = (len(train),1))))
    
    learning_rate = 0.005
    lambda_reg = 0.05
    # initiazlize random matrix
    X = np.random.randint(1,6,size = (train[:,0].max(),train[:,1].max()))
    U, s, M = np.linalg.svd(X,full_matrices = False)
    RMSE = []
    for iteration in range(5):
        predicted = np.empty((1,1))
        for el in train:
            x_hat = np.dot(U[el[0] - 1,:],M[:,el[1] - 1])
            predicted = np.append(predicted,x_hat)
            e_ij = el[2] - x_hat
            U[el[0] - 1,:] = U[el[0] - 1,:] + learning_rate*(2*e_ij*M[:,el[1] - 1] - lambda_reg*U[el[0] - 1,:])
            M[:,el[1] - 1] = M[:,el[1] - 1] + learning_rate*(2*e_ij*U[el[0] - 1,:] - lambda_reg*M[:,el[1] - 1])
        
        predicted = np.delete(predicted,0)
        RMSE.append(np.sqrt(np.mean((train[:,2] - predicted)**2)))
        if ((iteration > 1) and (RMSE[iteration] >= RMSE[iteration - 1]) and (RMSE[iteration] >= RMSE[iteration - 2])):
            err_train[fold] = RMSE[iteration]
            break
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))    
               