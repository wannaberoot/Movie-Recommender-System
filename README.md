## Movie-Recommender-System
Movie recommender system based on the MovieLens 100K Dataset with implementing collaborating filtering (CF) algorithms.

# Purpose of The Project

The purpose of this project is to develop a movie recommender system based on the MovieLens 100K Dataset with implementing collaborating filtering (CF) algorithms. The ultimate goal is to predict the rating given to a movie by a specific user.

# What Is Collaborative Filtering?

Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.

It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user. It looks at the items they like and combines them to create a ranked list of suggestions.

There are many ways to decide which users are similar and combine their choices to create a list of recommendations.

# User-Based vs Item-Based Collaborative Filtering

The technique in the examples explained above, where the rating matrix is used to find similar users based on the ratings they give, is called user-based or user-user collaborative filtering. If you use the rating matrix to find similar items based on the ratings given to them by users, then the approach is called item-based or item-item collaborative filtering.

The two approaches are mathematically quite similar, but there is a conceptual difference between the two. Here’s how the two compare:

User-based: For a user U, with a set of similar users determined based on rating vectors consisting of given item ratings, the rating for an item I, which hasn’t been rated, is found by picking out N users from the similarity list who have rated the item I and calculating the rating based on these N ratings.

Item-based: For an item I, with a set of similar items determined based on rating vectors consisting of received user ratings, the rating by a user U, who hasn’t rated it, is found by picking out N items from the similarity list that have been rated by U and calculating the rating based on these N ratings.

# Pearson Correlation vs Cosine Similarity

The relation between Pearson’s correlation coefficient and Salton’s cosine measure is revealed based on the different possible values of the division of the L1-norm and the L2-norm of a vector. These different values yield a sheaf of increasingly straight lines which form together a cloud of points, being the investigated relation. The theoretical results are tested against the author co-citation relations among 24 informetricians for whom two matrices can be constructed, based on co-citations: the asymmetric occurrence matrix and the
symmetric co-citation matrix. Both examples completely confirm the theoretical results. The results enable us to specify an algorithm which provides a threshold value for the cosine above which none of the corresponding Pearson correlations would be negative. Using this threshold value can be expected to optimize the visualization of the vector space.

# Dataset Overview

MovieLens dataset collected by GroupLens Research. In particular, the MovieLens 100k dataset is a stable benchmark dataset with 100,000 ratings given by 943 users for 1682 movies, with each user having rated at least 20 movies.

This dataset consists of many files that contain information about the movies, the users, and the ratings given by users to the movies they have watched. The ones that are of interest are the following:

•	u.item: the list of movies
•	u.data: the list of ratings given by users

By merging these 2 files, we can obtain a dataframe as follows:

![---](/images/1.png)

First 5 Rows

# K-Nearest Neighbors

The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.
The k-nearest neighbors (KNN) algorithm is a simple, supervised machine learning algorithm that can be used to solve both classification and regression problems. It’s easy to implement and understand, but has a major drawback of becoming significantly slows as the size of that data in use grows.
KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).
In the case of classification and regression, we saw that choosing the right K for our data is done by trying several Ks and picking the one that works best.


## Experimental Results

# Optimal K Value

On the user-based and item-based approaches we wrote, we calculate Mean Absolute Error (MAE) for each value of k by changing the values of k and we can try to find the optimal value of k by graphing results.
Note: MAE can range from 0 to ∞. MAE is negatively-oriented scores: Lower values are better.


1.	Optimal K Value for User-Based Approach:

![---](/images/2.png)

As can be seen from the graph, we can use the value 80 for k. It’s the lowest.


2.	Optimal K Value for Item-Based Approach:

![---](/images/3.png)

As can be seen from the graph, we can use the value 30, 70 or 80 for k but I will use 30 because of the lower k value is better for performance.

# Predicton

The goal of the project was to predict the rating given to a movie by a specific user. I run my prediction algorithms, which I have written separately for item-based and user-based approaches, with optimal k values by giving movie_id and user_id values.

![---](/images/4.png)

My user-based algorithm finds the rating predicton of movie_200 as “3” for user_1, while my item-based algorithm finds the same prediction as “4”.

![---](/images/5.png)

