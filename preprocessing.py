# %%
import pandas as pd
import numpy as np

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
df = pd.read_csv('cinemas.csv', index_col=0)
df.head()

# %%
df.info()

# %%
"""
## remove empty columns
"""

# %%
df['Unnamed: 28'].value_counts()

# %%
df.drop(columns=['Unnamed: 28'], inplace=True)

# %%
df.head()

# %%
"""
## cast columns
"""

# %%
cols = ['num_voted_users', 'facenumber_in_poster', 'num_user_for_reviews', 'title_year']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').astype('Int64')
categories = ['color', 'language', 'country', 'content_rating']
df[categories] = df[categories].astype('category')

# %%
df.info()

# %%
"""
## fill missing data
"""

# %%
df.war_symb_title.value_counts()

# %%
df.isna().sum()

# %%
total_missing = df.isna().sum()
percent_missing = (total_missing/df.shape[0]) * 100
percent_missing

# %%
fill_values = {
    'color': df['color'].mode()[0],
    'num_critic_for_reviews': df['num_critic_for_reviews'].mean(),
    'duration': df['duration'].mean(),
    'gross': df['gross'].median(),
    'facenumber_in_poster': df['facenumber_in_poster'].median(),
    'num_voted_users': df['num_voted_users'].median(),
    'director_name': df['director_name'].mode()[0],
    'actor_1_name': df['actor_1_name'].mode()[0],
    'actor_2_name': df['actor_2_name'].mode()[0],
    'actor_3_name': df['actor_3_name'].mode()[0],
    'director_facebook_likes': df['director_facebook_likes'].median(),
    'actor_1_facebook_likes': df['actor_1_facebook_likes'].median(),
    'actor_2_facebook_likes': df['actor_2_facebook_likes'].median(),
    'actor_3_facebook_likes': df['actor_3_facebook_likes'].median(),
    'plot_keywords': df['plot_keywords'].mode()[0],
    'num_user_for_reviews': df['num_user_for_reviews'].median(),
    'language': df['language'].mode()[0],
    'country': df['country'].mode()[0],
    'content_rating': df['content_rating'].mode()[0],
    'budget': df['budget'].median(),
    'title_year': df['title_year'].mode()[0],
    'aspect_ratio': df['aspect_ratio'].median()
}

df = df.fillna(fill_values)

# %%
df.isna().sum()

# %%
df = df.astype({
    'num_voted_users': 'int64',
    'title_year': 'int64'
})

# %%
"""
## remove duplicates
"""

# %%
df.duplicated().sum()

# %%
df[df.duplicated()].tail()

# %%
df = df.drop_duplicates()
df.duplicated().sum()

# %%
"""
## drop odd values
"""

# %%
for col in df.select_dtypes(include='category').columns:
    print(f"--- {col} ---")
    print(df[col].value_counts())

# %%
df = df.drop(df[df['color'] == 'Green and Yellow'].index)

# %%
df.loc[df['title_year'] == 200000]

# %%
df = df.drop(df[df['title_year'] == 200000].index)

# %%
df.describe(include='all')

# %%
"""
## one hot encoding
"""

# %%
for col in df.select_dtypes(include='category').columns:
    print(col)

# %%
df['color'] = df['color'].apply(lambda x: 1 if x == 'Color' else 0)

# %%
# one hot encoding
df = pd.get_dummies(df, columns=['language', 'country', 'content_rating'], drop_first=True, dtype=int, sparse=False)

# %%
df.iloc[:5, -10:]

# %%
"""
## one hot encoding multilabel columns
"""

# %%
from sklearn.preprocessing import MultiLabelBinarizer

df['genres_list'] = df['genres'].str.split('|')

mlb = MultiLabelBinarizer(sparse_output=False)
genres_encoded = pd.DataFrame(
    mlb.fit_transform(df['genres_list']),
    columns=[f'genres_{c}' for c in mlb.classes_],
    index=df.index
)

df = df.join(genres_encoded)
df = df.drop(columns=['genres_list'])

df.iloc[:5, -5:]

# %%
unique_genres = set(g for sublist in df['genres'].str.split('|') for g in sublist)
print("Number of unique genres:", len(unique_genres))

# %%
genre_sums = {col: df[col].sum() for col in df.columns if 'genres_' in col}
genre_sums_df = pd.DataFrame(list(genre_sums.items()), columns=['Genre', 'Count']).sort_values('Count', ascending=False).reset_index(drop=True)
genre_sums_df

# %%
genre_cols = [col for col in df.columns if col.startswith('genres_')]
genre_counts = df[genre_cols].sum()
cols_to_other = genre_counts[genre_counts < 100].index.tolist()
df['genres_Other'] = df[cols_to_other].max(axis=1)
df.drop(columns=cols_to_other, inplace=True)

# %%
x = [col for col in list(df.columns) if col.startswith('genres_')]
len(x)

# %%
genre_sums = {col: df[col].sum() for col in df.columns if 'genres_' in col}
genre_sums_df = pd.DataFrame(list(genre_sums.items()), columns=['Genre', 'Count']).sort_values('Count', ascending=False).reset_index(drop=True)
genre_sums_df

# %%
from collections import Counter
from itertools import chain

df['keywords_list'] = df['plot_keywords'].fillna('').str.split('|')

# select top 50 keywords
all_keywords = list(chain.from_iterable(df['keywords_list']))
top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(50)]
print("all keywords: ", len(all_keywords))
print("top keywords: ", top_keywords)

# replace rare keywords with "other"
df["keywords_filtered"] = df["keywords_list"].apply(
    lambda x: [kw if kw in top_keywords else "other" for kw in x if kw != '']
)

mlb = MultiLabelBinarizer(sparse_output=False)
keywords_encoded = mlb.fit_transform(df["keywords_filtered"])

keywords_df = pd.DataFrame(
    keywords_encoded,
    columns=[f"keywords_{c}" for c in mlb.classes_],
    index=df.index
)

df = df.join(keywords_df)
df.drop(columns=['keywords_list', 'keywords_filtered', 'plot_keywords', 'genres'], inplace=True)

print(df.iloc[:5, -10:])

# %%
keywords_stat = [(col, df[col].sum()) for col in df.columns if col.startswith('keywords_')]
keywords_stat_sorted = sorted(keywords_stat, key=lambda x: x[1], reverse=True)
keywords_stat_sorted

# %%
for i, col in enumerate(df.columns):
    print(f'{i+1}: {col}')

# %%
df.to_pickle("cinemas_cleaned.pkl")