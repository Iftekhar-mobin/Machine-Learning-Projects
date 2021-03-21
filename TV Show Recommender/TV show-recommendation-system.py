
# coding: utf-8

# # TVDB dataset
# 
# ## Data Exploration and TV Show Recommendation using Collaborative Filtering and Stochastic Gradient Descent

# ### How tvshows evolve over time?

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt; plt.rcdefaults()
from IPython import get_ipython
from sklearn.model_selection import train_test_split
import time

#start_time = time.clock()


tvshows = pd.read_csv('./Dataset/tvshows.csv')

#print(tvshows.head())

# Extract Title

# print(tvshows.shape)

tvshows.rename(columns={'title': 'title_name'}, inplace=True)

##print(tvshows.head())

plt.bar(list(tvshows.ix[0:50,'title_name']),list(tvshows.ix[0:50,'tvdbRating']), align='center', alpha=0.7)
plt.xticks(rotation=90)
plt.title("TV shows with their respective ratings")
plt.xlabel("TV Shows")
plt.ylabel("Average rating")
plt.show()


rating = pd.read_csv('./Dataset/ratings.csv')

rating.head()

rating.shape

# # #### Average tvshow rating

average_rating = rating.groupby('tvId')['rating'].mean()

average_rating.shape

average_rating = average_rating.reset_index()

tvshows_rating = pd.merge(tvshows, average_rating, how='left', on='tvId')

tvshows_rating.shape
tvshows_rating.head()

# # ## Collaborative Filtering Recommendation System
# #
# # In this section, we build a recommendation system based on the TV show rating data using a model-based Collaborative filtering approach, Matrix factorization, to predict the rating of the tvshows a user has not rated yet and to provide list of recommended tvshows based on those predicted rating.
# #
# # The idea behind matrix factorization is to learn latent factors associated with users and tvshows. We can think of the these latent factors as user's preferences and TV show characteristics respectively, which are hidden features but we can learn them from rating data given.

rating.shape
rating.head()

# # ### Number of reviews by user (distribution)
# # The chart below shows the distribution of the count of rating given by unique users. It illustrates that majority of users provided only few TV show ratings.

count_review_by_user = rating.groupby('userId')['rating'].count()
##print(count_review_by_user)
count_review_by_user.plot(kind='hist')
plt.show()

#
#
# # ### Number of reviews by tvshows (distribution)
# # The chart below shows the distribution of the count of ratings received by each TV show. It also illustrated that majority of the tvshows received rating only from few users.

count_review_by_tvshow = rating.groupby('tvId')['rating'].count()
##count_review_by_tvshow.plot(kind='hist')
##plt.show()
count_review_by_tvshow.describe()

max_count_review_by_tvshow = max(count_review_by_tvshow)
count_review_by_tvshow[count_review_by_tvshow==max_count_review_by_tvshow]
#print(tvshows[tvshows.tvId==most_rated_show.tvId])

# # ### Average rating by user and by Tv show:
# # ##### Average rating by user: for a user, what is the average TV show rating this user gave?
# # ##### Average rating by TV show: for a TV show, what is the average rating it has?

average_rating_by_user = rating.groupby('userId')['rating'].mean()
#average_rating_by_user.plot(kind='hist')
#plt.show()

average_rating_by_tvshow = rating.groupby('tvId')['rating'].mean()
# average_rating_by_tv.plot(kind='hist')
#plt.show()

# print ("Average rating by user = ", average_rating_by_user.mean())
# print ("Average rating by TV show = ", average_rating_by_TV show.mean())

# # Merge Review count and Average review to original dataframe
#count_review_by_user.shape, count_review_by_tvshow.shape, average_rating_by_user.shape, average_rating_by_tvshow.shape

count_review_by_user = count_review_by_user.reset_index()
count_review_by_tvshow = count_review_by_tvshow.reset_index()

average_rating_by_user = average_rating_by_user.reset_index()
average_rating_by_tvshow = average_rating_by_tvshow.reset_index()

count_review_by_user.columns = ['userId','count_rating_user']
count_review_by_tvshow.columns = ['tvId','count_rating_tvshow']

average_rating_by_user.columns = ['userId','average_rating_user']
average_rating_by_tvshow.columns = ['tvId','average_rating_tvshow']

df_rating = pd.merge(rating, count_review_by_user, how='left', on='userId')

df_rating = pd.merge(df_rating, count_review_by_tvshow, how='left', on='tvId')

df_rating = pd.merge(df_rating, average_rating_by_user, how='left', on='userId')

df_rating = pd.merge(df_rating, average_rating_by_tvshow, how='left', on='tvId')

#print(df_rating.head())

#print(df_rating.count_rating_user.describe())
#print(df_rating.count_rating_tvshow.describe())

# # ### Prepare dataset for Collaborative filtering
# #
# # In doing the collaborative filtering matrix factorization, we first need to construct our rating dataset into a users-tvshows matrix.

# #### We will work with small subset of rating data
#rating_subset = df_rating[(df_rating.count_rating_user >= 1500) & (df_rating.count_rating_tvshow >= 5000)]
rating_subset = df_rating[(df_rating.count_rating_user >= 300) & (df_rating.count_rating_tvshow >= 600)]
#rating_subset = df_rating

print(rating_subset.head())
#print(rating.shape)
#print(rating.head())
#
#

# len(rating_subset.userId.unique()), len(rating_subset.TV showId.unique())

# # Initiate number of users and tvshows
n_users = rating_subset.userId.unique().shape[0]
n_tvshows = rating_subset.tvId.unique().shape[0]

#print (n_users, n_tvshows)

# # We will split rating data into traning and testing datasets into 75:25 ratio

train_data, test_data = train_test_split(rating_subset, test_size=0.25)

train_data.shape, test_data.shape

len(train_data.userId.unique()), len(test_data.userId.unique()), len(train_data.tvId.unique()), len(test_data.tvId.unique())
#

train_data.head()

# # #### We organize the rating data into n x m matrix, where n is the number of users and m is the number of tvshows

# # Create training and test matrix using Pandas pivot
ptrain = train_data.reset_index().pivot_table(index='userId', columns='tvId', values='rating')

print(ptrain.head())

# # Store Index of userId
train_user_index = pd.DataFrame(ptrain.index)


# # Store Index of tvId
train_tvshows_index = pd.DataFrame(ptrain.columns)


ptest = test_data.reset_index().pivot_table(index='userId', columns='tvId', values='rating')

#ptest.head()

# # Convert to numpy matrix and fill NA with 0

fillna = 0
# # Copy to new dataframe representing matrices
R = ptrain.fillna(0.0).copy().values
T = ptest.fillna(0.0).copy().values

# # User-tvshows rating matrix (for training) is R
#print (R)

# # User-tvshows rating matrix (for testing) is T
#print (T)

# # Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# # Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# # ### Stochastic Gradient Descent
#
# # We initialize latent factors of users as P vector and latent factors of tvshows as Q vector
# # To predict rating of user i on TV show j

# # And we can work this out as a linear regression problem by minimizing cost function to learn parameter vectors P and Q

# # Following the cost (objective) function above, we can write stochastic gradient descent algorithm as:
# # 1. Intitalize P and Q vector of length k with random values (k = number of latent features)
# # 2. For each data point, we calculate gradient and update weight P and Q simultaneously until converged or reach maximum iteration

# # Predict the ratings through the dot product of the latent features for users and tvshows
def prediction(P,Q):
    return np.dot(P.T,Q)

lmbda = 0.1 # L2 penalty Regularization weight (Lambda)
k = 20  # number of the latent features
m, n = R.shape  # Number of users and tvshows
n_iter = 100  # Number of epochs
step_size = 0.01  # Learning rate or Step size (gamma)

P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent TV show feature matrix

### # Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R > 0]))

R.nonzero()

R.shape

Q.shape, P.shape

train_errors = []
test_errors = []

# #Only consider non-zero matrix
users, items = R.nonzero()
for iter in range(n_iter):
    for u, i in zip(users, items):
         e = R[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
         P[:,u] += step_size * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
         Q[:,i] += step_size * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent tvshow feature matrix
    train_rmse = rmse(I,R,Q,P) # Calculate root mean squared error from train dataset
    test_rmse = rmse(I2,T,Q,P) # Calculate root mean squared error from test dataset
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

plt.plot(range(n_iter), train_errors, marker='o', label='Training Data');
plt.plot(range(n_iter), test_errors, marker='v', label='Test Data');
plt.title('SGD-WR Learning Curve')
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.grid()
plt.show()

# # Calculate prediction matrix R_hat (low-rank approximation for R)
R = pd.DataFrame(R)
R_hat=pd.DataFrame(prediction(P,Q))

### # Compare true ratings of a user with predictions
ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
ratings.columns = ['Actual Rating', 'Predicted Rating']
print(ratings)

P.shape, Q.shape

R.loc[16,:].head()

R_hat.loc[16,:].head()

# # Calculate prediction matrix T_hat (low-rank approximation for T)
T = pd.DataFrame(T)
T_hat=pd.DataFrame(prediction(P,Q))

T.loc[16,:].head()

T_hat.loc[16,:].head()

train_user_index.loc[16,:].values[0]

train_data[train_data.userId==train_user_index.loc[16,:].values[0]].count()

# # Rank rating prediction for this user
predicted_ratings = pd.DataFrame(data=R_hat.loc[16,R.loc[16,:] == 0])

predicted_ratings.shape

top_10_reco = predicted_ratings.sort_values(by=16,ascending=False).head(10)

top_10_reco

top_10_reco.index.tolist()

# # List of 10 recommended tvshows for this user
print(tvshows[tvshows.tvId.isin(top_10_reco.index.tolist())])

train_user_index.iloc[16,:]

# # Top 10 tvshows this user had rated
pd.merge(rating[rating.userId==3907].sort_values(by='rating',ascending=False).head(10), tvshows, on='tvId', how='left')

#print (time.clock() - start_time, "seconds")
