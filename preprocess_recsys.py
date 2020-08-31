#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os

logging.basicConfig(level=logging.INFO)


if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    os.chdir("recsys")


# %% [markdown]
# ## Loading Data

# %% [markdown]
# We start by running the `download_and_process_data` function.
# The function returns the `df_train`, `df_test`, `df_dev`, `df_valid` dataframes, which correspond to our training, test, development, and validation sets.
# Each of those dataframes has the following fields:
# * `user_idx`: A unique identifier for a user.
# * `book_idx`: A unique identifier for a book that is being rated by the user.
# * `book_idxs`: The set of books that the user has interacted with (read or planned to read).
# * `review_text`: Optional text review written by the user for the book.
# * `rating`: Either `0` (which means the user did not read or did not like the book) or `1` (which means the user read and liked the book). The `rating` field is missing for `df_train`.
# Our objective is to predict whether a given user (represented by the set of book_idxs the user has interacted with) will read and like any given book.
# That is, we want to train a model that takes a set of `book_idxs` (the user) and a single `book_idx` (the book to rate) and predicts the `rating`.
#
# In addition, `download_and_process_data` also returns the `df_books` dataframe, which contains one row per book, along with metadata for that book (such as `title` and `first_author`).

# %% {"tags": ["md-exclude-output"]}
from utils_rec import download_and_process_data

(df_train, df_test, df_dev, df_valid), df_books = download_and_process_data()

df_books.head()

# %% [markdown]
# We look at a sample of the labeled development set.
# As an example, we want our final recommendations model to be able to predict that a user who has interacted with `book_idxs` (25743, 22318, 7662, 6857, 83, 14495, 30664, ...) would either not read or not like the book with `book_idx` 22764 (first row), while a user who has interacted with `book_idxs` (3880, 18078, 9092, 29933, 1511, 8560, ...) would read and like the book with `book_idx` 3181 (second row).

# %%
df_dev.sample(frac=1, random_state=12).head()

# %% [markdown]
# ## Writing Labeling Functions

# %%
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

# %% [markdown]
# If a user has interacted with several books written by an author, there is a good chance that the user will read and like other books by the same author.
# We express this as a labeling function, using the `first_author` field in the `df_books` dataframe.
# We picked the threshold 15 by plotting histograms and running error analysis using the dev set.

# %%
from snorkel.labeling.lf import labeling_function

book_to_first_author = dict(zip(df_books.book_idx, df_books.first_author))
first_author_to_books_df = df_books.groupby("first_author")[["book_idx"]].agg(set)
first_author_to_books = dict(
    zip(first_author_to_books_df.index, first_author_to_books_df.book_idx)
)


@labeling_function(
    resources=dict(
        book_to_first_author=book_to_first_author,
        first_author_to_books=first_author_to_books,
    )
)
def shared_first_author(x, book_to_first_author, first_author_to_books):
    author = book_to_first_author[x.book_idx]
    same_author_books = first_author_to_books[author]
    num_read = len(set(x.book_idxs).intersection(same_author_books))
    return POSITIVE if num_read > 15 else ABSTAIN


# %% [markdown]
# We can also leverage the long text reviews written by users to guess whether they liked or disliked a book.
# For example, the third `df_dev` entry above has a review with the text `'4.5 STARS'`, which indicates that the user liked the book.
# We write a simple LF that looks for similar phrases to guess the user's rating of a book.
# We interpret >= 4 stars to indicate a positive rating, while < 4 stars is negative.

# %%
low_rating_strs = [
    "one star",
    "1 star",
    "two star",
    "2 star",
    "3 star",
    "three star",
    "3.5 star",
    "2.5 star",
    "1 out of 5",
    "2 out of 5",
    "3 out of 5",
]
high_rating_strs = ["5 stars", "five stars", "four stars", "4 stars", "4.5 stars"]


# In[2]:


@labeling_function(
    resources=dict(low_rating_strs=low_rating_strs, high_rating_strs=high_rating_strs)
)
def stars_in_review(x, low_rating_strs, high_rating_strs):
    if not isinstance(x.review_text, str):
        return ABSTAIN
    for low_rating_str in low_rating_strs:
        if low_rating_str in x.review_text.lower():
            return NEGATIVE
    for high_rating_str in high_rating_strs:
        if high_rating_str in x.review_text.lower():
            return POSITIVE
    return ABSTAIN


# %% [markdown]
# We can also run [TextBlob](https://textblob.readthedocs.io/en/dev/index.html), a tool that provides a pretrained sentiment analyzer, on the reviews, and use its polarity and subjectivity scores to estimate the user's rating for the book.
# As usual, these thresholds were picked by analyzing the score distributions and running error analysis.

# %%
from snorkel.preprocess import preprocessor
from textblob import TextBlob


@preprocessor(memoize=True)
def textblob_polarity(x):
    if isinstance(x.review_text, str):
        x.blob = TextBlob(x.review_text)
    else:
        x.blob = None
    return x


# Label high polarity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def polarity_positive(x):
    if x.blob:
        if x.blob.polarity > 0.3:
            return POSITIVE
    return ABSTAIN


# Label high subjectivity reviews as positive.
@labeling_function(pre=[textblob_polarity])
def subjectivity_positive(x):
    if x.blob:
        if x.blob.subjectivity > 0.75:
            return POSITIVE
    return ABSTAIN


# Label low polarity reviews as negative.
@labeling_function(pre=[textblob_polarity])
def polarity_negative(x):
    if x.blob:
        if x.blob.polarity < 0.0:
            return NEGATIVE
    return ABSTAIN


# %% {"tags": ["md-exclude-output"]}
from snorkel.labeling import PandasLFApplier, LFAnalysis

lfs = [
    stars_in_review,
    shared_first_author,
    polarity_positive,
    subjectivity_positive,
    polarity_negative,
]

applier = PandasLFApplier(lfs)
#L_dev = applier.apply(df_dev)
L_train = applier.apply(df=df_train)
L_dev = applier.apply(df=df_dev)
L_valid = applier.apply(df=df_valid)
L_test = applier.apply(df=df_test)


# In[ ]:


# %% [markdown]
# ### Applying labeling functions to the training set
#
# We apply the labeling functions to the training set, and then filter out data points unlabeled by any LF to form our final training set.

# %% {"tags": ["md-exclude-output"]}
from snorkel.labeling.model import LabelModel

L_train = applier.apply(df_train)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)
preds_train = label_model.predict(L_train)

from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
    df_train, preds_train, L_train
)
df_train_filtered["rating"] = preds_train_filtered


# In[44]:


path="./Data/rec"
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer


# In[30]:


# df_train_filtered.review_text = df_train_filtered.review_text.fillna('no text')
# df_dev.review_text = df_dev.review_text.fillna('no text')


# In[31]:


# df_valid.review_text = df_valid.review_text.fillna('no text')
# df_test.review_text = df_test.review_text.fillna('no text')


# In[33]:


#vectorizer.transform(df_train_filtered[0:10].book_idxs)
#vectorizer = CountVectorizer()#ngram_range=(1, 2),max_features=10000)
# vectorizer = DictVectorizer()
#X_train = vectorizer.fit_transform(df_train_filtered.book_idxs)
#X_dev = vectorizer.transform(df_dev.book_idxs).toarray()
#X_valid = vectorizer.transform(df_valid.book_idxs.tolist())
#X_test = vectorizer.transform(df_test.book_idxs.tolist())


# In[35]:


#============ our changes ==================#
def lsnork_to_l_m(lsnork, num_classes):
	m = 1 - np.equal(lsnork,-1).astype(int)
	l = m*lsnork + (1-m)*num_classes
	return l,m


def get_features(df):
    t = df.book_idxs.values
    u = 200#[len(i) for i in t]
    v = [np.pad(i,(0,max(u)-len(i)),'constant') for i in t]
    return np.asarray(v)
import pickle
d_x = get_features(df_dev)
# d_x = df_dev.book_idxs.values#.toarray()
d_L = df_dev.rating.values
d_l = L_dev

d_l, d_m = lsnork_to_l_m(d_l,2)
d_d = np.array([1.0] * len(d_x))
d_r = np.zeros(d_l.shape) #rule exemplar coupling unavailable


with open(path+"/"+"d_processed.p","wb") as f:
    pickle.dump(d_x,f)
    pickle.dump(d_l,f)
    pickle.dump(d_m,f)
    pickle.dump(d_L,f)
    pickle.dump(d_d,f)
    pickle.dump(d_r,f)


# In[121]:


# U_x = X_train.toarray()
U_x = get_features(df_train_filtered)# toarray()
U_L = df_train_filtered.rating.values
U_l = L_train
U_l, U_m = lsnork_to_l_m(U_l,2)
U_d = np.array([0.0] * len(U_x))
U_r = np.zeros(U_l.shape)

with open(path+"/"+"U_processed.p","wb") as f:
    pickle.dump(U_x,f)
    pickle.dump(U_l,f)
    pickle.dump(U_m,f)
    pickle.dump(U_L,f)
    pickle.dump(U_d,f)
    pickle.dump(U_r,f)


# In[122]:



valid_x = get_features(df_valid)
valid_L = df_valid.rating.values
valid_l = L_valid
valid_l, valid_m = lsnork_to_l_m(valid_l,2)
valid_d = np.array([0.0] * len(valid_x))
valid_r = np.zeros(valid_l.shape) #rule exemplar coupling unavailable
with open(path+"/"+"validation_processed.p","wb") as f:
	pickle.dump(valid_x,f)
	pickle.dump(valid_l,f)
	pickle.dump(valid_m,f)
	pickle.dump(valid_L,f)
	pickle.dump(valid_d,f)
	pickle.dump(valid_r,f)


test_x = get_features(df_test)
test_L = df_test.rating.values
test_l = L_test
test_l, test_m = lsnork_to_l_m(test_l,2)
test_d = np.array([0.0] * len(test_x))
test_r = np.zeros(test_l.shape) #rule exemplar coupling unavailable
with open(path+"/"+"test_processed.p","wb") as f:
	pickle.dump(test_x,f)
	pickle.dump(test_l,f)
	pickle.dump(test_m,f)
	pickle.dump(test_L,f)
	pickle.dump(test_d,f)
	pickle.dump(test_r,f)




exit()


# In[11]:


LFAnalysis(L_dev, lfs).lf_summary(df_dev.rating.values)


# In[ ]:


# %% [markdown]
# ### Applying labeling functions to the training set
#
# We apply the labeling functions to the training set, and then filter out data points unlabeled by any LF to form our final training set.

# %% {"t


# In[12]:


# %% [markdown]
# ### Applying labeling functions to the training set
#
# We apply the labeling functions to the training set, and then filter out data points unlabeled by any LF to form our final training set.

# %% {"tags": ["md-exclude-output"]}
from snorkel.labeling.model import LabelModel

L_train = applier.apply(df_train)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train, n_epochs=5000, seed=123, log_freq=20, lr=0.01)
preds_train = label_model.predict(L_train)

from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
    df_train, preds_train, L_train
)
df_train_filtered["rating"] = preds_train_filtered


# In[27]:


df_train_filtered.head()


# In[ ]:




