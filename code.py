# import the numpy module as np
import numpy as np

# load data
ratings = np.genfromtxt("ml-1m/ratings.dat",usecols = (0,1,2),delimiter = "::",dtype = 'int')

# indicate the number of folds we are going to use for cross validation
nfolds = 5
# set random seed for reproducibility of results
np.random.seed(17)

# create a sequence of indices for each of the 5 folds 
seqs = [x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)


def UserAverage():
    
    # initialize the vectors that will contain the errors for each fold
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    
    # number of users in the data set
    num_users = max(ratings[:,0])
    
    # initialize a 5-dimensional array that will contain the mean ratings for each fold
    predictions_perUser_5dim = np.empty((nfolds,1,num_users))
    # initialize two 5-dimensional arrays that will contain the indices of 
    # every new user id, for each fold, for the train and test set
    num_ratings_perUser_train_5dim = np.empty((nfolds,1,num_users + 1),dtype = int)
    num_ratings_perUser_test_5dim = np.empty((nfolds,1,num_users + 1),dtype = int)
    # for each fold:
    for fold in range(nfolds):
        train_index = np.array([x != fold for x in seqs])
        test_index = np.array([x == fold for x in seqs])
        # train set of current fold
        train = ratings[train_index]
        # test set of current fold
        test = ratings[test_index] 
        
        # number of ratings per user for the train and test set 
        num_ratings_perUser_train = np.bincount(train[:,0])
        num_ratings_perUser_test = np.bincount(test[:,0])
        num_ratings_perUser_train_5dim[fold] = num_ratings_perUser_train
        num_ratings_perUser_test_5dim[fold] = num_ratings_perUser_test
        
        # the cumulative sum indicates the index in which every new user (user_id) 
        # appears in the train and test set. The sets are sorted by user id.
        index_perUser_train = np.cumsum(num_ratings_perUser_train)
        index_perUser_test = np.cumsum(num_ratings_perUser_test)
    
        unique_userIDs = set(train[:,0])
        # initialize an array that will contain the prediction per user for the train and test set
        predictions_perUser_train = np.empty(len(train))
        predictions_perUser_test = np.empty(len(test))
        # initialize an array that will contain the unique predictions per user
        unique_predictions_perUser = np.empty(num_users)
        
        # for each user
        for user_id in range(num_users):
            # calculate the mean value of the movies per user
            if (user_id + 1) in unique_userIDs:
                unique_predictions_perUser[user_id] = np.mean(train[index_perUser_train[user_id]:index_perUser_train[user_id + 1],2])
                # or calculate the global average (fall back value), in case
                # the user is not contained in the train set (all ratings of the
                # user are in the test set)
            else:
                unique_predictions_perUser[user_id] = np.mean(train[:,2])
            # repeat the calculated mean values for each row for the same user id, since the prediction is
            # just the mean value of all the movies a specific user has rated
            predictions_perUser_train[index_perUser_train[user_id]:index_perUser_train[user_id + 1]] = np.repeat(
                    unique_predictions_perUser[user_id],num_ratings_perUser_train[user_id + 1])
            predictions_perUser_test[index_perUser_test[user_id]:index_perUser_test[user_id + 1]] = np.repeat(
                    unique_predictions_perUser[user_id],num_ratings_perUser_test[user_id + 1])
        
        # round the predictions. It is guranteed that the predictions/mean values
        # are between [1,5], since the mean value of numbers in the range [1,5] 
        # can not be greater than 5 or less than 1
        predictions_perUser_train = np.around(predictions_perUser_train)
        predictions_perUser_test = np.around(predictions_perUser_test)
        unique_predictions_perUser = np.around(unique_predictions_perUser)
        predictions_perUser_5dim[fold] = unique_predictions_perUser
        
        # calculate the Root Mean Squared Error for the train and test set    
        RMSE_train[fold] = np.sqrt(np.mean((train[:,2] - predictions_perUser_train)**2))
        RMSE_test[fold] = np.sqrt(np.mean((test[:,2] - predictions_perUser_test)**2))
        
        # print the RMSEs for the train and test set (for each fold)
        print("Fold " + str(fold) + ": RMSE_train=" + str(RMSE_train[fold]) + "; RMSE_test=" + str(RMSE_test[fold]))
    
    # Create three attributes of the UserAverage function, containing the predictions for each fold,
    # and the number of ratings per user in the train and test set. These values will be used
    # in the LeastSquares() function
    UserAverage.ratings = predictions_perUser_5dim
    UserAverage.num_ratings_perUser_train = num_ratings_perUser_train_5dim
    UserAverage.num_ratings_perUser_test = num_ratings_perUser_test_5dim
    
    # print the average error of the 5 folds, for the train and test set. 
    print("\n")
    print("Mean RMSE on the train set: " + str(np.mean(RMSE_train)))
    print("Mean RMSE on the test set: " + str(np.mean(RMSE_test)))
    
    

def ItemAverage():
    
    # initialize the vectors that will contain the errors for each fold
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    
    # number of movies in the data set
    num_movies = max(ratings[:,1])

    # initialize a 5-dimensional array that will contain the mean ratings for each fold
    predictions_perItem_5dim = np.empty((nfolds,1,num_movies))
    # initialize two 5-dimensional arrays that will contain the indices of 
    # every new item id, for each fold, for the train and test set
    num_ratings_perItem_train_5dim = np.empty((nfolds,1,num_movies + 1),dtype = int)
    num_ratings_perItem_test_5dim = np.empty((nfolds,1,num_movies + 1),dtype = int)
    # for each fold:
    for fold in range(nfolds):
        train_index = np.array([x != fold for x in seqs])
        test_index = np.array([x == fold for x in seqs])
        # train set of current fold
        train = ratings[train_index]
        # test set of current fold
        test = ratings[test_index] 
    
        # sort the train and test set by item id
        train = train[train[:,1].argsort()]
        test = test[test[:,1].argsort()]
        # number of ratings per item for the train and test set 
        num_ratings_perItem_train = np.bincount(train[:,1])
        num_ratings_perItem_test = np.bincount(test[:,1])
        num_ratings_perItem_train_5dim[fold] = num_ratings_perItem_train
        num_ratings_perItem_test_5dim[fold] = num_ratings_perItem_test
        
        # the cumulative sum indicates the indices in which every new item (item_id) 
        # appears in the train and test set. The sets are already sorted by item id.
        index_perItem_train = np.cumsum(num_ratings_perItem_train)
        index_perItem_test = np.cumsum(num_ratings_perItem_test)
    
        unique_itemIDs = set(train[:,1])
        # Initialize two arrays that will contain the predictions per item for the train and test set
        predictions_perItem_train = np.empty(len(train))
        predictions_perItem_test = np.empty(len(test))
        # initialize an array that will contain the unique predictions per item
        unique_predictions_perItem = np.empty(num_movies)
        
        # for each item
        for item_id in range(num_movies):
            # calculate the mean values of the movies per item
            if (item_id + 1) in unique_itemIDs:
                unique_predictions_perItem[item_id] = np.mean(train[index_perItem_train[item_id]:index_perItem_train[item_id + 1],2])
                # or calculate the global average (fall back value), in case
                # the item is not contained in the train set (all ratings of the
                # item are in the test set)
            else:
                unique_predictions_perItem[item_id] = np.mean(train[:,2])
                # repeat the calculated mean values for each row for the same item id
                # in order to create two arrays containing the mean ratings for each
                # available combination of user and item, with length equal to that 
                # of the train set and test set, respectively
            predictions_perItem_train[index_perItem_train[item_id]:index_perItem_train[item_id + 1]] = np.repeat(
                    unique_predictions_perItem[item_id],num_ratings_perItem_train[item_id + 1])
            predictions_perItem_test[index_perItem_test[item_id]:index_perItem_test[item_id + 1]] = np.repeat(
                    unique_predictions_perItem[item_id],num_ratings_perItem_test[item_id + 1])
    
        # round the predictions
        predictions_perItem_train = np.around(predictions_perItem_train)
        predictions_perItem_test = np.around(predictions_perItem_test)
        unique_predictions_perItem = np.around(unique_predictions_perItem)
        predictions_perItem_5dim[fold] = unique_predictions_perItem
        
        # calculate the Root Mean Squared Error for the train and test set    
        RMSE_train[fold] = np.sqrt(np.mean((train[:,2] - predictions_perItem_train)**2))
        RMSE_test[fold] = np.sqrt(np.mean((test[:,2] - predictions_perItem_test)**2))

        # print the RMSEs for the train and test set (for each fold)
        print("Fold " + str(fold) + ": RMSE_train=" + str(RMSE_train[fold]) + "; RMSE_test=" + str(RMSE_test[fold]))
        
    ItemAverage.ratings = predictions_perItem_5dim
    ItemAverage.num_ratings_perItem_train = num_ratings_perItem_train_5dim
    ItemAverage.num_ratings_perItem_test = num_ratings_perItem_test_5dim    
    # print the average error of the 5 folds, for the train and test sets. 
    print("\n")
    print("Mean RMSE on the train set: " + str(np.mean(RMSE_train)))
    print("Mean RMSE on the test set: " + str(np.mean(RMSE_test)))
    
def LeastSquares():
    
    # initialize the arrays that will contain the errors for each fold
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    # number of users and movies/items
    num_users = max(ratings[:,0])
    num_movies = max(ratings[:,1]) 
    
    # for each fold:
    for fold in range(nfolds):
        train_index = np.array([x != fold for x in seqs])
        test_index = np.array([x == fold for x in seqs])
        train = ratings[train_index] 
        test = ratings[test_index] 
        
        # Use the attributes of the UserAverage() function to extract the number of ratings per user
        # and then calculate the indices of every new user id that appears in the train and test set
        num_ratings_perUser_train = UserAverage.num_ratings_perUser_train[fold][0]
        num_ratings_perUser_test = UserAverage.num_ratings_perUser_test[fold][0]
        index_perUser_train = np.cumsum(num_ratings_perUser_train)
        index_perUser_test = np.cumsum(num_ratings_perUser_test)
        # Use the attributes of the ItemAverage() function to extract the number of ratings per item
        # and then calculate the indices of every new item id that appears in the train and test set
        num_ratings_perItem_train = ItemAverage.num_ratings_perItem_train[fold][0]
        num_ratings_perItem_test = ItemAverage.num_ratings_perItem_test[fold][0]
        index_perItem_train = np.cumsum(num_ratings_perItem_train)
        index_perItem_test = np.cumsum(num_ratings_perItem_test)    
        
        # initialize the arrays that will contain the predictions per user for the train and test set
        predictions_perUser_train = np.empty(len(train))
        predictions_perUser_test = np.empty(len(test))
        unique_predictions_perUser = UserAverage.ratings[fold][0]
        for user_id in range(num_users):
            predictions_perUser_train[index_perUser_train[user_id]:index_perUser_train[user_id + 1]] = np.repeat(
                    unique_predictions_perUser[user_id],num_ratings_perUser_train[user_id + 1])
            predictions_perUser_test[index_perUser_test[user_id]:index_perUser_test[user_id + 1]] = np.repeat(
                    unique_predictions_perUser[user_id],num_ratings_perUser_test[user_id + 1])   
        
        # initialize the arrays that will contain the predictions per item for the train and test set
        predictions_perItem_train = np.empty(len(train))
        predictions_perItem_test = np.empty(len(test))
        unique_predictions_perItem = UserAverage.ratings[fold][0]
        for item_id in range(num_movies):
            predictions_perItem_train[index_perItem_train[item_id]:index_perItem_train[item_id + 1]] = np.repeat(
                    unique_predictions_perItem[item_id],num_ratings_perItem_train[item_id + 1])
            predictions_perItem_test[index_perItem_test[item_id]:index_perItem_test[item_id + 1]] = np.repeat(
                    unique_predictions_perItem[item_id],num_ratings_perItem_test[item_id + 1])       
        
        # calculate parameters a,b,c using the least squares method
        A = np.vstack((predictions_perUser_train,predictions_perItem_train,np.ones(len(train)))).T
        a,b,c = np.linalg.lstsq(A,train[:,2])[0] 
        
        # calculate the Root Mean Squared Error for the train and test set 
        RMSE_train[fold] = np.sqrt(np.mean((train[:,2] - (predictions_perUser_train*a + predictions_perItem_train*b + c))**2))
        RMSE_test[fold] = np.sqrt(np.mean((test[:,2] - (predictions_perUser_test*a + predictions_perItem_test*b + c))**2))
    
        # print the RMSEs for the train and test set (for each fold)
        print("Fold " + str(fold) + ": RMSE_train=" + str(RMSE_train[fold]) + "; RMSE_test=" + str(RMSE_test[fold]))

    # print the average error of the 5 folds, for the train and test sets. 
    print("\n")
    print("Mean error on the train set: " + str(np.mean(RMSE_train)))
    print("Mean error on the test set: " + str(np.mean(RMSE_test)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    







