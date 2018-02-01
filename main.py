import argparse
import numpy as np

def MatrixFactorization(data, delimiter = None, useCols = 'all', dtype = None,
                        method = None, percTrain = 0.7, numFolds = None, log = True,
                        logIteration = False, seed = 100, learningRate = 0.005, lambdaReg = 0.05, 
                        numFeatures = 10, iterations = 75):

    # check type of input
    if isinstance(data, np.ndarray):
        if useCols != 'all':
            data = data[:, useCols]
    elif isinstance(data, list):
        data = np.asarray(data)
        if useCols != 'all':
            data = data[:, useCols]
    else:
        if useCols == 'all':
            data = np.genfromtxt(data, delimiter = delimiter, dtype = dtype)
        else:
            data = np.genfromtxt(data, delimiter = delimiter, usecols = useCols, dtype = dtype)
    
    np.random.seed(seed)

    # check type of method
    if method == None:
        rows = np.arange(len(data))
        np.random.shuffle(rows)
        trainIndex = rows[1:round(percTrain*len(data))]
        testIndex = rows[round(percTrain*len(data)):]
        train = data[trainIndex]
        test = data[testIndex]
    else:
        # create folds
        seqs = [x % numFolds for x in range(len(data))]
        np.random.shuffle(seqs)

    if method == None:
        numFolds = 1

    RMSEtrain = np.zeros(numFolds)
    RMSEtest = np.zeros(numFolds)
    MAEtrain = np.zeros(numFolds)
    MAEtest = np.zeros(numFolds)

    # for each fold
    for fold in range(numFolds):
        if method == 'cv':
            # create the train and test set
            trainIndex = np.array([x != fold for x in seqs])
            testIndex = np.array([x == fold for x in seqs])
            train = data[trainIndex]
            test = data[testIndex]
            
        # initiazlize matrices U and M 
        np.random.seed(seed + 50)
        U = np.random.rand(np.max(train[:, 0]), numFeatures)
        M = np.random.rand(numFeatures, np.max(train[:, 1]))
        
        RMSElist = []
        MAElist = []
        if log == True:
            if method == 'cv':
                print("")
                print(str(numFolds) + "-Fold Cross Validation: Fold " + str(fold + 1))
                print("-------------------------------")
        
        # for each iteration:
        for iteration in range(iterations):
            SSEtrain = 0
            SAEtrain = 0
            
            # for each record in the train set
            for idx, rating in enumerate(train):
                u = U[rating[0] - 1,:].copy()
                # calculate the rating (prediction) the user would give to the movie
                prediction = np.dot(u,M[:,rating[1] - 1])
                # supress the rating between 1 and 5
                if prediction < 1:
                    prediction = 1
                elif prediction > 5:
                    prediction = 5
                error = rating[2] - prediction
                SSEtrain += error**2
                SAEtrain += abs(error)

                # update matrices U and M
                U[rating[0] - 1, :] += learningRate*(2*error*M[:, rating[1] - 1] - lambdaReg*u)
                M[: ,rating[1] - 1] += learningRate*(2*error*u - lambdaReg*M[:,rating[1] - 1])
            RMSEiter = np.sqrt(SSEtrain / len(train))
            MAEiter = SAEtrain / len(train)

            if log == True:
                print("")
                print("Root Mean Squared Error (RMSE) for iteration " + str(iteration + 1) + " on train set: " + str(RMSEiter))
                print("Mean Absolute Error (MAE) for iteration " + str(iteration + 1) + " on train set: " + str(MAEiter))
                print("")
                
            RMSElist.append(RMSEiter)
            MAElist.append(MAEiter)
        RMSEtrain[fold] = RMSElist[-1]
        MAEtrain[fold] = MAElist[-1]
        
        numUsers = max(train[:, 0])
        numRatingsPerUserTest = np.bincount(test[:, 0])
        indexPerUserTest = np.cumsum(numRatingsPerUserTest)

        # make predictions on the test set
        SSEtest = 0
        SAEtest = 0
        for userID in range(numUsers):
            testSubset = test[indexPerUserTest[userID]:indexPerUserTest[userID + 1], :]
            predictions = np.dot(U[userID, :], M[:, testSubset[:, 1] - 1])
            SSEtest += np.sum((testSubset[:, 2] - predictions)**2)
            SAEtest += np.sum(abs(testSubset[:,2] - predictions))
        RMSEtest[fold] = np.sqrt(SSEtest/len(test))
        MAEtest[fold] = SAEtest/len(test)

        if log == True:
            if method == 'cv':
                print("Fold " + str(fold + 1) + ": Root Mean Squared Error (RMSE) on train set: " + str(RMSEtrain[fold]))
                print("Fold " + str(fold + 1) + ": Root Mean Squared Error (RMSE) on test set: " + str(RMSEtest[fold]))
                print("")

                print("Fold " + str(fold + 1) + ": Mean Absolute Error (MAE) on train set: " + str(MAEtrain[fold]))
                print("Fold " + str(fold + 1) + ": Mean Absolute Error (MAE) on test set: " + str(MAEtest[fold]))
                print("")
            else:
                print("Root Mean Squared Error (RMSE) on train set: " + str(RMSEtrain[fold]))
                print("Root Mean Squared Error (RMSE) on test set: " + str(RMSEtest[fold]))
                print("")

                print("Mean Absolute Error (MAE) on train set: " + str(MAEtrain[fold]))
                print("Mean Absolute Error (MAE) on test set: " + str(MAEtest[fold]))
                print("")

    if log == True:
        if method == 'cv':
            print("Mean of Root Mean Squared Error (RMSE) on train sets (over " + str(numFolds) + "folds): " + str(np.mean(RMSEtrain)))
            print("Mean of Root Mean Squared Error (RMSE) on test sets (over" + str(numFolds) + " folds): " + str(np.mean(RMSEtest)))

            print("Mean of Mean Absolute Error (MAE) on train sets (over " + str(numFolds) + "folds): " + str(np.mean(MAEtrain)))
            print("Mean of Mean Absolute Error (MAE) on test sets (over " + str(numFolds) + "folds): " + + str(np.mean(MAEtest)))
    
    # matrix X contains the predicted rating a user (row) gave to a movie (column)
    X = np.dot(U, M)
    return(X, U, M, RMSEtest, MAEtest)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-delimiter', '-d', default = None)
    parser.add_argument('-useCols', '-uc', default = None)
    parser.add_argument('-dtype', '-dt', default = None)
    parser.add_argument('-method', '-m', default = None)
    parser.add_argument('-percTrain', '-pt', default = 0.7, type = float)
    parser.add_argument('-numFolds', '-nf', default = None, type = int)
    parser.add_argument('-log', '-l', default = 'True')
    parser.add_argument('-logIteration', '-li', default = 'True')
    parser.add_argument('-seed', '-s', default = 100, type = int)
    parser.add_argument('-learningRate', '-lr', default = 0.005, type = float)
    parser.add_argument('-lambdaReg', '-lreg', default = 0.05, type = float)
    parser.add_argument('-numFeatures', '-fe', default = 10, type = int)
    parser.add_argument('-iterations', '-i', default = 75, type = int)
    args = parser.parse_args()

    MatrixFactorization(args.data, args.delimiter, eval(args.useCols), args.dtype,
                        args.method, args.percTrain, args.numFolds, bool(args.log),
                        bool(args.logIteration), args.seed, args.learningRate, args.lambdaReg, 
                        args.numFeatures, args.iterations)
    