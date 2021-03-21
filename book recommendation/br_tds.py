import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
#import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import warnings

# Reading the data
book = pd.read_csv('Bx-Books.csv', sep=';',error_bad_lines=False, encoding="latin-1")
book.columns=['ISBN','bookTitle','bookAuthor','yearOfPublication','publisher','imageUrlS','imageUrlM','imageUrlL']
user=pd.read_csv('BX-Users.csv', sep=';',error_bad_lines=False, encoding="latin-1")
user.columns=['userID','Location','Age']
rating=pd.read_csv('BX-Book-Ratings.csv', sep=';',error_bad_lines=False, encoding="latin-1")
rating.columns=['userID','ISBN','bookRating']

# No. of ratings
print(rating.shape)
print(list(rating.columns))
rating.head()

# Checking the ratings distribution
plt.rc("font", size=15)
rating.bookRating.value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating') 
plt.ylabel('Count')
plt.savefig('system1.png',bbox_inches='tight')
plt.show()

# No. of books
print(book.shape)
print(list(book.columns))
book.head()

# No. of users
print(user.shape)
print(list(user.columns))
user.head()

# Count of users based on age
user.Age.hist(bins=[0,10,20,30,40,50,100])
plt.title('Age Distribution\n')
plt.xlabel('Age') 
plt.ylabel('Count')
plt.savefig('system2.png',bbox_inches='tight')
plt.show()

# Grouping data based on Rating Count
rating_count = pd.DataFrame(rating.groupby('ISBN')['bookRating'].count())
rating_count.sort_values('bookRating', ascending=False).head()

# Grouping data using Pearsons' co-relation coefficient
average_rating = pd.DataFrame(rating.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = rating_count
average_rating.sort_values('ratingCount', ascending=False).head()

# Excluding data for statistical significance
# Excluding all users with less than 200 ratings
count1 = rating['userID'].value_counts()
rating = rating[rating['userID'].isin(count1[count1>=200].index)]

# Excluding all books with less than 100 ratings
count2 = rating['bookRating'].value_counts()
rating = rating[rating['bookRating'].isin(count2[count2>=100].index)]

# Creating a 2D matrix containing userId and their rating of each book
rating_pivot = rating.pivot(index='userID',columns='ISBN').bookRating
userID = rating_pivot.index
ISBN = rating_pivot.columns
print(rating_pivot.shape)
rating_pivot.head()

# Recommending books based on the book "The Lovely Bones: A Novel"
bones_rating = rating_pivot['0316666343']
similar_to_bones = rating_pivot.corrwith(bones_rating)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR',ascending=False).head(10)

books_corr_to_bones = pd.DataFrame(corr_summary.loc[:,'ISBN'])
corr_books = pd.merge(books_corr_to_bones, book, on='ISBN')
print(corr_books.shape)

# Combining book data and rating data
combine_book_rating = pd.merge(rating,book,on='ISBN')
columns=['yearOfPublication','publisher','bookAuthor','imageUrlS','imageUrlM','imageUrlL']
combine_book_rating=combine_book_rating.drop(columns,axis=1)
combine_book_rating.head()
combine_book_rating=combine_book_rating.dropna(axis=0,subset=['bookTitle'])

# Creating a new column for total rating count
book_ratingCount=(combine_book_rating.groupby(by=['bookTitle'])['bookRating'].count().reset_index().rename(columns={'bookRating':'totalRatingCount'})[['bookTitle','totalRatingCount']])
book_ratingCount.head()

# Combining rating data with total rating count
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
rating_with_totalRatingCount.head()

# Count of total rating count
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['totalRatingCount'].describe())

# Limiting to books which received 50 or more ratings
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

# Limiting user data based on location: US and Canada
combined = rating_popular_book.merge(user, left_on = 'userID', right_on = 'userID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
print(us_canada_user_rating.head())

# Implementing kNN
us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(us_canada_user_rating_matrix)

# Testing the kNN algorithm 
query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))

#SVD- Matrix Factorization model
us_canada_user_rating_pivot2 = us_canada_user_rating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)
us_canada_user_rating_pivot2.head()

print(us_canada_user_rating_pivot2.shape)

X = us_canada_user_rating_pivot2.values.T
print(X.shape)

# Using in-built compression model, the dimension of the data has been reduced
SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
print(matrix.shape)

#Calculation of Pearson's Correlation Co-efficient for every book pair
warnings.filterwarnings("ignore",category =RuntimeWarning)
corr = np.corrcoef(matrix)
print(corr.shape)

us_canada_book_title = us_canada_user_rating_pivot2.columns
us_canada_book_list = list(us_canada_book_title)
coffey_hands = us_canada_book_list.index("The Green Mile: Coffey's Hands (Green Mile Series)")
print(coffey_hands)
corr_coffey_hands = corr[coffey_hands]
list(us_canada_book_title[(corr_coffey_hands<1.0) & (corr_coffey_hands>0.9)])
