### Matrix Factorization for Recommendation: An Implementation using Netflix data.

This is a project in the framework of the course *Advances in Data Mining* of Leiden University. 

Recommendation systems are software tools and techniques [1] used in order to filter massive amounts of information [2] and recommend specific products or items to users that are highly likely to like, and therefore give a high rating. Thus, they are utilized by many commercial areas including, but not limited to movies, news, books, music etc. Many different methods exist for constructing a recommender system such as naive approaches, in which the system calculates the average rating of an item as rated by different users, or calculates the average rating of the items by the same user, and then recommends an item that has a relatively high average rating. In more advanced methods such as collaborative filtering [1], [2], [3], [4], [5], the same items are recommended to "similar" users, these are users that tend to like the same items, and have common preferences. Another approach, called content based approach [1], [2], [3], [6], items are recommended to a user because he/she liked similar items in the past. Finally, other approaches include matrix factorization such as, for example, Singular Value Decomposition [7], [8].

This project is a Python implementation of the Matrix Factorization technique described in []. The function **MatrixFactorization()** takes several arguments, specifically:     

* **data** (*string*): Directory of thhe data on which the technique is applied.
* **delimiter** (*string*), *Default* **None**: Delimiter symbol to split the columns of the data.
* **useCols** (*tuple*), *Default* **None**: Columns to be chosen from the data set, in order to apply the technique. In the case of Netflix data, these columns would be `user_id`, `movie_id`, `rating`.
