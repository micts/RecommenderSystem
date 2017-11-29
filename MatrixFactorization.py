def MatrixFactorization():
    
    # import the numpy module as np
    import numpy as np
    import time
   
    # load data
    ratings = np.genfromtxt("ratings.dat",usecols = (0,1,2),delimiter = "::",dtype = 'int')
    # indicate the number of folds we are going to use for cross validation
    nfolds = 5
    # set random seed for reproducibility of results
    np.random.seed(17)

    # create a sequence of indices for each of the 5 folds 
    seqs = [x%nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)
    
    # start timer
    start_func_timer = time.time()   
    # initialize the arrays that will contain the errors for each fold
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    MAE_train = np.zeros(nfolds)
    MAE_test = np.zeros(nfolds)
    RMSE_all = np.empty((5,1,75))
    MAE_all = np.empty((5,1,75))
    # for each fold
    for fold in range(nfolds):
        # start fold timer
        start_fold_timer = time.time()
        # create the train and test set
        train_index = np.array([x != fold for x in seqs])
        test_index = np.array([x == fold for x in seqs])
        train = ratings[train_index] 
        test = ratings[test_index]
            
        # set the learning rate and the lambda coefficient
        learning_rate = 0.005
        lambda_reg = 0.05
        
        # initiazlize matrices U and M and also set the seed in order to
        # initialize the same matrices for each fold
        np.random.seed(4)
        U = np.random.rand(max(train[:,0]),10)
        M = np.random.rand(10,max(train[:,1]))
        # initialize two lists that will contain the RMSE and MAE of each iteration, respectively
        RMSE_list = []
        MAE_list = []
        # for each iteration:
        for iteration in range(75):
            # Sum of Squared Errors
            SSE_train = 0
            # Sum of Absolute Errors
            SAE_train = 0
            # print current fold and current iteration
            print("Fold: " + str(fold + 1) + "," + " Iteration: " + str(iteration + 1))
            # for each record in the train set 
            for idx,rating in enumerate(train):
                # create a copy of the user vector 
                u = U[rating[0] - 1,:].copy()
                # calculate the rating (prediction) the user would give to the movie 
                x_hat = np.dot(u,M[:,rating[1] - 1])
                # supress the rating between 1 and 5
                if x_hat < 1:
                    x_hat = 1
                elif x_hat > 5:
                    x_hat = 5
                # calculate the error
                e_ij = rating[2] - x_hat
                # add the error to the Sum of Squared Errors
                SSE_train += e_ij**2
                # add the error the Sum of Absolute Errors
                SAE_train += abs(e_ij)
                
                # update matrices U and M
                U[rating[0] - 1,:] += learning_rate*(2*e_ij*M[:,rating[1] - 1] - lambda_reg*u) 
                M[:,rating[1] - 1] += learning_rate*(2*e_ij*u - lambda_reg*M[:,rating[1] - 1])
            # calculate the RMSE and MAE of this iteration, respectively    
            RMSE_iteration = np.sqrt(SSE_train/len(train))
            MAE_iteration = SAE_train/len(train)
            print("Root Mean Squared Error (RMSE) for iteration " + str(iteration + 1) + ": " + str(RMSE_iteration))
            print("Mean Absolute Error (MAE) for iteration " + str(iteration + 1) + ": " + str(MAE_iteration))
            print("")
            # Add/append the errors to the lists containing the errors of previous iterations
            RMSE_list.append(RMSE_iteration)
            MAE_list.append(MAE_iteration)
        # RMSE_all and MAE_all contain the list of errors of all iterations for the current fold.
        RMSE_all[fold] = RMSE_list
        MAE_all[fold] = MAE_list
        # the RMSE and MAE of the current fold are the last calculated RMSE and MAE, respectively    
        RMSE_train[fold] = RMSE_list[-1]
        MAE_train[fold] = MAE_list[-1]
            
        # calculate the number of users
        num_users = max(train[:,0])
        # calculate the number of times a user appears in the test set
        num_ratings_perUser_test = np.bincount(test[:,0])
        # the cumulative sum indicates the index in which every new user (user_id) 
        # appears in the test set.
        index_perUser_test = np.cumsum(num_ratings_perUser_test)
            
        # evaluate the model on the test set (make predictions on the test set)
        SSE_test = 0
        SAE_test = 0
        for user_id in range(num_users):
            test_subset = test[index_perUser_test[user_id]:index_perUser_test[user_id + 1],:]
            predictions = np.dot(U[user_id,:],M[:,test_subset[:,1] - 1])
            SSE_test += np.sum((test_subset[:,2] - predictions)**2)
            SAE_test += np.sum(abs(test_subset[:,2] - predictions))
        # calculate the RMSE and MAE of the test set  
        RMSE_test[fold] = np.sqrt(SSE_test/len(test))   
        MAE_test[fold] = SAE_test/len(test)
        
        # stop fold timer
        end_fold_timer = time.time()
        # print how much time taken (in minutes) for the fold to be executed. 
        # Also print the RMSE and MAE of the train and test fold 
        print("Time taken for fold " + str(fold + 1) + "(in minutes):" + str((end_fold_timer - start_fold_timer)/60))
        print("Fold " + str(fold + 1) + ": Root Mean Squared Error (RMSE) on train set: " + str(RMSE_train[fold]) +
              "; Root Mean Squared Error (RMSE) on test set: " + str(RMSE_test[fold]))
        print("Fold " + str(fold + 1) + ": Mean Absolute Error (MAE) on train set: " + str(MAE_train[fold]) + 
              "; Mean Absolute Error (MAE) on test set: " + str(MAE_test[fold]))
        print("")
    
    # print the average RMSE of the 5 folds, for the train and test sets. 
    print("Mean of Root Mean Squared Error (RMSE) on train sets: " + str(np.mean(RMSE_train)))
    print("Mean of Root Mean Squared Error (RMSE) on test sets: " + str(np.mean(RMSE_test)))
    
    # print the average MAE of the 5 folds, for the train and test sets. 
    print("Mean of Mean Absolute Error (MAE) on train sets: " + str(np.mean(MAE_train)))
    print("Mean of Mean Absolute Error (MAE) on test sets: " + str(np.mean(MAE_test)))
    
    # end timer
    end_func_timer = time.time()
    # print how much time (in hours) took for the function to be evaluated
    print("Time taken to evaluate function (in hours): " + str(((end_func_timer - start_func_timer)/60)/60))
    
    # return matrix U and M, RMSE and MAE for each iteration of the 5 folds, and finally
    # the RMSE and MAE of the 5 test folds
    return(U,M,RMSE_all,MAE_all,RMSE_test,MAE_test)
    
    