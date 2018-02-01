## Matrix Factorization for Recommendation: An Implementation using Netflix data.

This is a project in the framework of the course *Advances in Data Mining* of Leiden University. 

### Introduction

Recommendation systems are software tools and techniques [1] used in order to filter massive amounts of information [2] and recommend specific products or items to users that are highly likely to like, and therefore give a high rating. Thus, they are utilized by many commercial areas including, but not limited to movies, news, books, music etc. Many different methods exist for constructing a recommender system such as naive approaches, in which the system calculates the average rating of an item as rated by different users, or calculates the average rating of the items by the same user, and then recommends an item that has a relatively high average rating. In more advanced methods such as collaborative filtering [1], [2], [3], [4], [5], the same items are recommended to "similar" users, these are users that tend to like the same items, and have common preferences. Another approach, called content based approach [1], [2], [3], [6], items are recommended to a user because he/she liked similar items in the past. Finally, other approaches include matrix factorization such as, for example, Singular Value Decomposition [7], [8].

### Matrix Factorization

This project is a Python implementation of the Matrix Factorization technique described in [7]. In matrix factorization, the goal is to estimate matrix <a href="https://www.codecogs.com/eqnedit.php?latex=X&space;\in&space;\mathbb{R}^{I\times&space;J}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X&space;\in&space;\mathbb{R}^{I\times&space;J}" title="X \in \mathbb{R}^{I\times J}" /></a> containing the ratings given by a user <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> to a movie <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a>, using a matrix decomposition method, called Singular Value Decomposition (SVD). If we are able to predict the rating a user would give to a movie, then if the rating is higher than a specific threshold, this movie would be recommended to the user. Using SVD, the approximation of matrix <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a> can be written as a product of two matrices, namely <img src="https://latex.codecogs.com/gif.latex?U&space;\in&space;\mathbb{R}^{I&space;\times&space;K}" title="U \in \mathbb{R}^{I \times K}" />, and <a href="https://www.codecogs.com/eqnedit.php?latex=M&space;\in&space;\mathbb{R}^{K&space;\times&space;J}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M&space;\in&space;\mathbb{R}^{K&space;\times&space;J}" title="M \in \mathbb{R}^{K \times J}" /></a>, that is <a href="https://www.codecogs.com/eqnedit.php?latex=X\simeq&space;U\cdot&space;M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X\simeq&space;U\cdot&space;M" title="X\simeq U\cdot M" /></a>. Matrices <a href="https://www.codecogs.com/eqnedit.php?latex=U" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U" title="U" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M" title="M" /></a> contain <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> features for each user and movie. For example, a feature of matrix <a href="https://www.codecogs.com/eqnedit.php?latex=U" target="_blank"><img src="https://latex.codecogs.com/gif.latex?U" title="U" /></a> could be how much a specific user <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> likes action movies, and the corresponding feature of matrix <a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M" title="M" /></a> could be how much a specific movie <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> is considered to be an action movie. The dot product of all these different feature vectors for each available combination of user and movie yields our estimate of the rating a user <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> would give to the movie <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a>. High values of both these two feature vectors yield higher ratings, whereas the opposite is true for small values in both of these two feature vectors. It should be noted that we do not actually know what these feature vectors represent. The goal is to find the values of these vectors that best represent/estimate matrix <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a>, without knowing the actual meaning of them.

You can also check the (sligthly different) `C++` implementation in Simon's Funk blog [here](http://sifter.org/~simon/journal/20061211.html). This Python implementation is based on the one described in [7]. 

### Data

We use the *MovieLens 1M* [9] data set which can be found [here](http://grouplens.org/datasets/movielens/). The data set contains about 1.000.000 ratings given to about 4.000 movies by about 6.000 users. Additionally, some information is provided about movies (genre, title, production year) and users (gender, age, occupation). We are only interested in the `ratings.dat` file, which is of the from `<user_id`, `movie_id`, `rating`, `timestamp>`, and more specifically, we are not interested in the timestamp.   

### Short Documentation and Example

The function **MatrixFactorization()** performs Cross Validation (CV) if specified by the user, for a given number of folds defined by `numFolds`. The default is no CV, in which case, the data are splitted into train and test sets, where the percent of train instances is defined by `percTrain`. Default value is 0.7. Furthermore, the user can specify the `learningRate`, the `lambdaReg` which is the regularization parameter, the number of features `numFeatures`, and the number of iterations `iterations`.    

For more details, see [7].

More specifically, the function accepts the following input arguments:     

* **data** (*string*): Directory of the data on which the technique is applied.    
* **delimiter** **(-d)**, (*string*), *Default* **None**: Delimiter symbol to split the columns of the data.     
* **useCols** **(-uc)** (*tuple*), *Default* **None**: Columns to be chosen from the data set, in order to apply the technique. In the case of Netflix data, these columns would be `user_id`, `movie_id`, `rating`.     
* **dtype** **(-dt)** (*string*), *Default* **None**: Type of the data in columns (for example 'int').     
* **method** **(-m)** (*string*), *Default* **None**: Whether to perform Cross Validation (CV). Input 'cv' to perfrom CV.    
* **percTrain** **(-pt)** (*float*), *Default* **None**: Percent of train set, in case no CV is performed.    
* **numFolds** **(-nf)** (*int*), *Default* **None**: Number of folds for CV.    
* **log** **(-l)** (*boolean*), *Default* **True**: Whether to print an informative log.    
* **logIteration** **(-li)** (*boolean*), *Default* **True**: Whether to print an informative log in each iteration.
* **seed** **(-s)** (*int*), *Default* **100**: Seed for reproducibility of results.
* **learningRate** **(-lr)** (*float*), *Default* **0.005**: The learning rate.
* **lambdaReg** **(-lreg)** (*float*), *Default* **0.05**: Regularization parameter to avoid overfitting.    
* **numFeatures** **(-fe)** (*int*), *Default* **10**: The number of features for users and movies.    
* **iterations** **(-i)** (*int*), *Default* **75**: The number of iterations the algorithm should perform.

To run from terminal:    

```
python main.py ratings.dat -d :: -uc (0,1,2) -dt int -m cv -nf 5 -l True -li True -s 100 -lr 0.005 -lreg 0.05 -fe 10 -i 75    
```    
Within Python (in this example for no CV, where we split train/test into 80/20):   
```python
import main
main.MatrixFactorization(ratings.dat', 
                         delimiter = "::", 
                         useCols = (0,1,2), 
                         dtype = int, 
                         percTrain = 0.8, 
                         seed = 150, 
                         learningRate = 0.003, 
                         lambdaReg = 0.04, 
                         numFeatures = 15, 
                         iterations = 60)

```



### References   
[1] Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In *Recommender systems handbook*                           (pp. 1-35). springer US.    
[2] Isinkaye, F. O., Folajimi, Y. O., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. *Egyptian Informatics Journal*, 16(3), 261-273.     
[3] Leskovec, J., Rajaraman, A., & Ullman, J. D. (2014). *Mining of massive datasets*. Cambridge university press.     
[4] Schafer, J. B., Frankowski, D., Herlocker, J., & Sen, S. (2007). Collaborative filtering recommender systems. In *The adaptive web* (pp. 291-324). Springer, Berlin, Heidelberg.     
[5] Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001, April). Item-based collaborative filtering recommendation algorithms. In *Proceedings of the 10th international conference on World Wide Web* (pp. 285-295). ACM.     
[6] Brusilovski, P., Kobsa, A., & Nejdl, W. (Eds.). (2007). *The adaptive web: methods and strategies of web personalization* (Vol. 4321). Springer Science & Business Media.    
[7] Liu, B., Bennett, J., Elkan, C., Smyth, P., & Tikk, D. (2007, August). KDD Cup and Workshop 2007. In *Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining* (p. 2). ACM.    
[8] Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2000). *Application of dimensionality reduction in recommender system-a case study* (No. TR-00-043). Minnesota Univ Minneapolis Dept of Computer Science.     
[9] Harper, F. M., & Konstan, J. A. (2016). The movielens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems* (TiiS), 5(4), 19.     




