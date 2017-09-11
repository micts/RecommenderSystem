
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
    
# calculate the mean rating for the movies of each user
    avg_movie_ratings2 = np.zeros(train[:,0].max())
    for user_id in range(train[:,0].max()):
        if (user_id + 1) in ratings[:,0]:
            if (user_id + 1) in train[:,0]: # use global average ratings in case all instances of a user are in the test set
                #avg_movie_ratings[user_id] = np.mean(train[train[:,0] == user_id + 1][:,2])
                avg_movie_ratings2[user_id] = np.mean(train[np.where(train[:,0] == user_id + 1)][:,2])
            else:
                avg_movie_ratings2[user_id] = np.mean(train[:,2])
    
    # calculate train and test error
    SE_train = 0 # squared error for train
    for el in train:
        SE_train = SE_train + (el[2] - avg_movie_ratings2[el[0] - 1])**2
    err_train[fold] = np.sqrt(SE_train/len(train))
    
    SE_test = 0 # squared error for test
    for el in test:
        SE_test = SE_test + (el[2] - avg_movie_ratings2[el[0] - 1])**2    
    err_test[fold] = np.sqrt(SE_test/len(test))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

#print the final conclusion:
print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))













