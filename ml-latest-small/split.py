import pandas as pd
import pdb

ratings = pd.read_csv('ratings.csv')
def train_test_split(ratings):
    test = set(range(len(ratings))[::10])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]

ratings_train, ratings_test = train_test_split(ratings)
ratings_train.to_csv('ratings_train.csv', index=False)
ratings_test.to_csv('ratings_test_truth.csv', index=False)

ratings_test.loc[:, 'rating'] = 0
ratings_test.loc[:, 'timestamp'] = 0
ratings_test.to_csv('ratings_test.csv', index=False)
