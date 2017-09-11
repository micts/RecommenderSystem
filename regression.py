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
    
    # calculate the mean rating of each movie (item)
    avg_movie_ratings = np.zeros(train[:,1].max())
    for movie_id in range(train[:,1].max()):
        if (movie_id + 1) in ratings[:,1]: # many movie ids are missing from the original data set
            if (movie_id + 1) in train[:,1]: # use global average ratings in case all movie ids are in the test set
                avg_movie_ratings[movie_id] = np.mean(train[train[:,1] == movie_id + 1][:,2])
            else:
                avg_movie_ratings[movie_id] = np.mean(train[:,2])
    
    # calculate the mean rating for the movies of each user
    avg_movie_ratings2 = np.zeros(train[:,0].max())
    for user_id in range(train[:,0].max()):
        if (user_id + 1) in ratings[:,0]:
            if (user_id + 1) in train[:,0]: # use global average ratings in case all instances of a user are in the test set
                #avg_movie_ratings[user_id] = np.mean(train[train[:,0] == user_id + 1][:,2])
                avg_movie_ratings2[user_id] = np.mean(train[np.where(train[:,0] == user_id + 1)][:,2])
            else:
                avg_movie_ratings2[user_id] = np.mean(train[:,2])
                
    item_ratings_train = np.empty((1,1))
    user_ratings_train = np.empty((1,1))
    for el in range(train[:,0].max()):
        if el <= train[:,1].max() - 1:
            item_ratings_train = np.append(item_ratings_train,np.repeat(avg_movie_ratings[el],len(train[train[:,1] == el + 1])))
        user_ratings_train = np.append(user_ratings_train,np.repeat(avg_movie_ratings2[el],len(train[train[:,0] == el + 1])))    
    item_ratings_train = np.delete(item_ratings_train,0)
    user_ratings_train = np.delete(user_ratings_train,0)
        
    A = np.vstack((user_ratings_train,item_ratings_train,np.ones(len(train)))).T
    a,b,c = np.linalg.lstsq(A,train[:,2])[0]    
                
    err_train[fold] = np.sqrt(np.mean((train[:,2] - (user_ratings_train*a + item_ratings_train*b + c))**2))
    
    item_ratings_test = np.empty((1,1))
    user_ratings_test = np.empty((1,1))
    for el in range(test[:,0].max()):
        if el <= test[:,1].max() - 1:
            item_ratings_test = np.append(item_ratings_test,np.repeat(avg_movie_ratings[el],len(test[test[:,1] == el + 1])))
        user_ratings_test = np.append(user_ratings_test,np.repeat(avg_movie_ratings2[el],len(test[test[:,0] == el + 1])))    
    item_ratings_test = np.delete(item_ratings_test,0)
    user_ratings_test = np.delete(user_ratings_test,0)
    
    err_test[fold] = np.sqrt(np.mean((test[:,2] - (user_ratings_test*a + item_ratings_test*b + c))**2))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

#print the final conclusion:
print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))









