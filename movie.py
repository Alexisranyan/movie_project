
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from dython import nominal


OMdb = pd.read_csv('data/OMdb_mojo_clean.csv', na_values = ['NaN', 'inf'])
tmdb = pd.read_csv('data/tmdb_5000_movies.csv', na_values = ['NaN', 'inf'])
all_movie = pd.read_csv('data/all_movie.csv', na_values = ['NaN', 'inf'])

# From all_movie.csv, merge Writer1 through Writer4 into OMdb_mojo_clean.csv
#for matching movies by “Title” in all_movie.csv
movie_writers = all_movie.loc[:, (all_movie.columns.str.startswith('Write')) | (all_movie.columns == 'Title')]
OMdb = OMdb.merge(movie_writers, how = 'left', left_on = 'Title', right_on = 'Title')
OMdb.columns

#From tmdb_5000_movies.csv merge “budget” field into OMdb_mojo_clean.csv by
# matching movie “title”
tmdb_new = tmdb[['title', 'budget']]
OMdb = OMdb.merge(tmdb_new, how = 'left', left_on = 'Title', right_on = 'title')
OMdb.drop(columns = ['title'], inplace = True)

#From Cast1 to Cast 6 all_movie.csv merge into OMdb_mojo_clean.csv by
#matching “Title”
movie_casts = all_movie.loc[:, (all_movie.columns.str.startswith('Cast')) | (all_movie.columns == 'Title')]
OMdb = OMdb.merge(movie_casts, how = 'left', left_on = 'Title', right_on = 'Title')
OMdb = OMdb.drop_duplicates(keep = 'first')

# Data Preprocessing
# In OMdb_mojo.clean.csv, clean up the “nan” and “inf” and set them to 0
OMdb._get_numeric_data().isnull().sum().sort_values(ascending = False).head(10)
OMdb.loc[OMdb['BoxOffice'].isnull(), 'BoxOffice'] = 0
OMdb.loc[OMdb['logBoxOffice'].isnull(), 'logBoxOffice'] = 0
OMdb.loc[OMdb['budget'].isnull(), 'budget'] = 0
OMdb.loc[OMdb['overseas-gross'].isnull(), 'overseas-gross'] = 0
OMdb.loc[OMdb['bo_year_rank'].isnull(), 'bo_year_rank'] = 0
OMdb.loc[OMdb['domestic-gross'].isnull(), 'domestic-gross'] = 0
OMdb._get_numeric_data().isnull().sum().sort_values(ascending = False).head(10)
OMdb = OMdb.fillna('0')
OMdb.isnull().sum()
OMdb = OMdb[(OMdb[['budget', 'worldwide-gross']] != 0).all(axis =1)]
OMdb.shape

#Perform hot encoding for all the non-numeric data columns in OMdb_mojo_clean.csv, need to have the data ready to be fed to sci-kit library calls for logistic regression, KNN, #SVM etc.
cat_cols = np.array(pd.DataFrame(OMdb.dtypes[OMdb.dtypes == 'object']).index)
ohe = OneHotEncoder(drop = 'first')
ohe_array = ohe.fit_transform(OMdb[cat_cols]).toarray()
ohe_OMdb = pd.DataFrame(ohe_array, index = OMdb.index, columns = ohe.get_feature_names())
ohe_OMdb.head()
OMdb_drop_col = OMdb.drop(columns = cat_cols)
OMdb_ohed = pd.concat([OMdb_drop_col, ohe_OMdb], axis = 1)
OMdb_ohed.head()

'''
 Need correlation matrix for at least these features:
 
    Awards Director Genre IMdb_score Production Rated Writer 4 Writer 3
    Runtime actor_1
    actor_2 worldwide-gross studios oscar_noms oscar_wins writer2
    overseas-gross awards director_1 director_2 imdb_votes nomination writer1 language
'''
corr_df = OMdb[['Awards', 'Runtime', 'overseas-gross', 'Director', 'actor_1','awards', 'Genre', 'actor_2','director_1', 'IMdb_score', 'worldwide-gross', 'director_2', 'Production', 'studio','imdbVotes','Rated', 'oscar_noms', 'nominations','Writer 4', 'oscar_wins', 'Writer 1','Writer 3', 'Writer 2', 'Language']]
nominal.associations(corr_df, nominal_columns = 'all', figsize=(15, 15), annot =True)

#Covariance matrix to determine which features are similar and can be dropped from model input
OMdb.cov()

'''
EDA 
All graphs should be based on the OMdb_mojo_clean.csv (after the merges from the above requirements).
1. Graph #1 to Plot revenue by genre. Revenue comes from “worldwide-gross” and “Genre” in OMDb_mojo_clean.csv. Now some of the genre categories are multiples, you can count each combination as a unique category OR just pick one category in the list as the genre. So if a data point has Genre = Action, Comedy, Animation, we can pick “Action” or “Comedy” or “Animation” as its genre
'''
OMdb['genre_1'] = [i.split(',')[0] for i in OMdb['Genre'] ]
genre_revenue_df= pd.DataFrame(OMdb.groupby(['genre_1']).sum()['worldwide-gross'])
plt.figure(figsize = (10, 5))
plt.bar(genre_revenue_df.index, genre_revenue_df['worldwide-gross'])
plt.title('Revenue By Genre')
plt.xlabel('Genre')
plt.ylabel('Revenue \n (in 100 Billions)')
plt.show()


'''
2. Take all the movies and bin them by the month that they are released. The year of release we do not care about. Then plot them against the revenue generated per movie. So y-axis is revenue and x-axis is the months of the year. And each month of the year contains each movie.
'''
def add_month(df_col):
    months = []
    for data in df_col:
        if data != '0':
            months.append(pd.to_datetime(data).month)
        else:
            months.append(0)
    return (months)

OMdb['month'] = add_month(OMdb['Released'])
OMdb['month'] 
month_revenue_df =pd.DataFrame(OMdb.groupby(by = ['month', 'Title']).sum()['worldwide-gross'])
month_revenue_df.reset_index(inplace = True)
plt.scatter(month_revenue_df['month'], month_revenue_df['worldwide-gross'])
plt.title('Revernue Generated \n per Movie per Month')
plt.xlabel('Month')
plt.ylabel('Revenue \n (in Billions)')
plt.show()

'''
3. For the data points that have budget, revenue (Worldwide-gross) and director (pick director_1) filled in the merged OMDb_mojo_clean.csv..., can we calculate the percentage return on a movie? So it will be (budget/revenue) * 100% equals percentage return on a movie. I want to bin the movies by the director, so that we see which director has the highest percentage return on average. If you think a scatterplot would work better or some other chart, go ahead and do it.
'''
OMdb['pct_return'] = OMdb['budget']/OMdb['worldwide-gross']*100
revenue_director_df = pd.DataFrame(OMdb.groupby(by=['Director']).mean()['pct_return'])
revenue_director_df[revenue_director_df['pct_return']> 100]
plt.figure(figsize = (35,7))
plt.bar(revenue_director_df.index, revenue_director_df['pct_return'])
plt.title('Percentage Return on Average per Director', fontsize = 30)
plt.ylim(0,500)
plt.xlabel('Director', fontsize = 15)
plt.ylabel('Percentage Return', fontsize = 15)
plt.xticks(rotation = 90)
plt.show()

'''4. Can we make separate scatterplots (or whatever graphs you deem useful for visualization) for:
a. Worldwide-gross to production studio
b. Worldwide-gross to # of oscar wins
c. Worldwide-gross to # of oscar nomination
d. Worldwide-gross to total # awards won
e. Worldwide-gross to actor_1
f. Worldwide-gross to writer_1
'''
# a. Worldwide-gross to production studio
wwg_production_df = pd.DataFrame(OMdb.groupby(by=['Production']).sum()['worldwide-gross'])
wwg_studio_df = pd.DataFrame(OMdb.groupby(by=['studio']).sum()['worldwide-gross'])

plt.figure(figsize = (30,15))
ax1 = plt.subplot(2,1,1)
ax1.set_title('Revenue by Production', fontsize = 30)
ax1.bar(wwg_production_df.index, wwg_production_df['worldwide-gross'])
ax1.set_xlabel('Production', fontsize = 15)
ax1.set_xticklabels(wwg_production_df.index, rotation = 90)
ax1.set_ylabel('Revenue \n (in 10 Billions)', fontsize = 15)

ax2 = plt.subplot(2,1,2)
ax2.set_title('Revenue by Studio', fontsize = 30)
ax2.bar(wwg_studio_df.index, wwg_studio_df['worldwide-gross'])
ax2.set_xlabel('Studio', fontsize = 15)
ax2.set_xticklabels(wwg_studio_df.index, rotation = 90)
ax2.set_ylabel('Revenue \n (in 10 Billions)', fontsize = 15)

plt.tight_layout()
plt.show()

# b. Worldwide-gross to # of oscar wins
plt.scatter(OMdb['oscar_wins'], OMdb['worldwide-gross'])
plt.title('Worldwid-gross to number of Oscar wins')
plt.xlabel('Number of Oscar Wins')
plt.ylabel('Revenue \n (in Billions)')
plt.show()

# c. Worldwide-gross to # of oscar nominations
plt.scatter(OMdb['oscar_noms'], OMdb['worldwide-gross'])
plt.title('Worldwid-gross to number of Oscar wins')
plt.xlabel('Number of Oscar Nomination')
plt.ylabel('Revenue \n (in Billions)')
plt.show()

#d. Worldwide-gross to total # awards won
plt.scatter(OMdb['awards'], OMdb['worldwide-gross'])
plt.title('Worldwid-gross to number of Oscar wins')
plt.xlabel('Number of awards')
plt.ylabel('Revenue \n (in Billions)')
plt.show()

# Use log to reduce the heteroscedasticity
plt.scatter(np.log(OMdb['awards']), np.log(OMdb['worldwide-gross']))
plt.title('Worldwid-gross to number of Oscar wins \n after logarithm')
plt.xlabel('Natrual Log of the Number of awards')
plt.ylabel('Natural Log of the Revenue')
plt.show()

# Worldwide-gross to actor_1
wwg_actor1_df = pd.DataFrame(OMdb.groupby(by=['actor_1']).sum()['worldwide-gross'])
plt.figure(figsize = (35,7))
plt.bar(wwg_actor1_df.index, wwg_actor1_df['worldwide-gross'])
plt.title('Revenue by actor_1', fontsize = 30)
plt.xlabel('Actor Name', fontsize = 15)
plt.ylabel('Revenue \n (in 10 Billions)', fontsize = 15)
plt.xticks(rotation = 90)
plt.show()

# Worldwide-gross to writer_1
wwg_writer_df = pd.DataFrame(OMdb.groupby(by=['Writer 1']).sum()['worldwide-gross'])
plt.figure(figsize = (40,7))
plt.bar(wwg_writer_df.index, wwg_writer_df['worldwide-gross'])
plt.title('Revenue by Write_1', fontsize = 30)
plt.xlabel('Write Name', fontsize = 15)
plt.ylabel('Revenue \n (in 10 Billions)', fontsize = 15)
plt.xticks(rotation = 90)
plt.show()