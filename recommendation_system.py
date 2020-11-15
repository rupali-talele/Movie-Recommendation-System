from collections import Counter
import math
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy
import pdb
import itertools


def movies_with_genres(movies_file):
    movies = pd.read_csv(movies_file, encoding='utf-8')
    genre_list = []
    for index, movie in movies.iterrows():
        genre_list.append(movie.genres.split('|'))
    movies['genres_list'] = genre_list
    return movies


def get_genre_dict(movies):
    # create genre dictionary
    genre_list = movies["genres_list"].tolist()
    genres = set(genre for genres in genre_list for genre in genres)
    genre_dict = {genre: index for index,
                  genre in enumerate(sorted(list(genres)))}
    return genre_dict


def create_shingles(hash_size, movies, genre_dict):
    shingles = []
    movie_index_dict = {}
    for index, movie in movies.iterrows():
        shingle = set()
        for genre in movie['genres_list']:
            genre_hash = hash(genre) % hash_size
            
            shingle.add(genre_hash)
        movie_index_dict[movie['movieId']] = index
        shingles.append(shingle)

    return shingles, movie_index_dict


def get_hash_coeffs(br):
    rnds = np.random.choice(2**10, (2, br), replace=False)
    c = 1048583
    return rnds[0], rnds[1], c


def min_hashing(shingles, hash_coeffs, br):
    count = len(shingles)
    (a, b, c) = hash_coeffs
    a = a.reshape(1, -1)

    M = np.zeros((br, count), dtype=int)

    for i, s in enumerate(shingles):
        row_idx = np.asarray(list(s)).reshape(-1, 1)
        m = (np.matmul(row_idx, a) + b) % c
        m_min = np.min(m, axis=0)
        M[:, i] = m_min
    return M


def LSH(M, b, r, band_hash_size):
    count = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        row_idx = []
        col_idx = []
        row_start = band_index * r
        for c in range(count):
            v = M[row_start:(row_start+r), c]
            v_hash = hash(tuple(v.tolist())) % band_hash_size
            row_idx.append(v_hash)
            col_idx.append(c)

        data_ary = [True] * len(row_idx)

        m = scipy.sparse.csr_matrix((data_ary, (row_idx, col_idx)), shape=(
            band_hash_size, count), dtype=bool)
        bucket_list.append(m)

    return bucket_list

def find_candidates(
        movieidx, bucket_list, M, b, r, band_hash_size, movies_rated_by_user):

    candidates = set()
    movies_rated_by_user = set(movies_rated_by_user)
    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), movieidx]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash].indices
        # pdb.set_trace()
        bucket = movies_rated_by_user.intersection(bucket)
        candidates = candidates.union(bucket)       
    
    return candidates

def find_pair_similarity(shingles, movieid1, movieid2, candidates, threshold):
    shingle1 = shingles[movieid1]
    shingle2 = shingles[movieid2]
    sim = len(shingle1 & shingle2)/len(shingle1 | shingle2)
    if sim >= threshold:
        return sim
    return 0

def create_dictionaries(ratings_train, ratings_test, movie_index_dict):
    movies_rated_dict = {}
    ratings_rated_dict = {}
    ratings_test_unique = pd.DataFrame(
        {'userId': ratings_test.userId.unique()})
    
    for index, movie in ratings_test_unique.iterrows():
        if movie['userId'] not in movies_rated_dict.keys():
            movies = list(ratings_train.loc[ratings_train['userId'] == movie['userId']]['movieId'])
            movies_rated_by_user = [movie_index_dict[m] for m in movies]

            movies_rated_dict[movie['userId']] = movies_rated_by_user

            ratings_list = list(ratings_train.loc[ratings_train['userId'] == movie['userId']]['rating'])
            ratings_rated_dict[movie['userId']] = ratings_list

    return movies_rated_dict, ratings_rated_dict
    

def calculate_ratings(movies, ratings_train, ratings_test, threshold, bucket_list, M, b, r, band_hash_size, shingles, movie_index_dict):
    ratings = []

    movies_rated_dict, ratings_rated_dict = create_dictionaries(
        ratings_train, ratings_test,movie_index_dict)
    
    for index, movie in ratings_test.iterrows():

        movies_rated_by_user = movies_rated_dict[movie['userId']]
        
        ratings_list = ratings_rated_dict[movie['userId']]

        target_movie = movie_index_dict[movie['movieId']]

        candidates = find_candidates(
            target_movie, bucket_list, M, b, r, band_hash_size, movies_rated_by_user)

        similarity_of_movies = [0]*len(movies_rated_by_user)
        for i,mov in enumerate(movies_rated_by_user):
            if mov in candidates:
                similarity_of_movies[i] = find_pair_similarity(shingles, target_movie, mov, candidates, threshold)

        weighted_ratings = sum([sim*ratings_list[i] for i, sim in enumerate(similarity_of_movies) if sim>0])

        above_threshold = [
            sim for sim in similarity_of_movies if sim>0]

        if len(above_threshold) == 0:
            ratings.append(np.mean(ratings_list))
        else:
            ratings.append(weighted_ratings/sum(above_threshold))

    return np.array(ratings)


# python rupali_vinayak_talele_task2.py ml-latest-small\movies.csv ml-latest-small\ratings_train.csv ml-latest-small\ratings_test.csv output_task2.csv

if __name__ == '__main__':
    time_start = time.time()
    movies_file = sys.argv[1]
    ratings_train_file = sys.argv[2]
    ratings_test_file = sys.argv[3]
    output_file = sys.argv[4]

    movies = movies_with_genres(movies_file)

    ratings_test = pd.read_csv(ratings_test_file,encoding='utf-8')
    ratings_train = pd.read_csv(ratings_train_file, encoding='utf-8')
    

    genre_dict = get_genre_dict(movies)

    hash_size = 2**20
    shingles, movie_index_dict = create_shingles(
        hash_size, movies, genre_dict)
    b = 45
    r = 2
    br = b*r
    band_hash_size = 2**16
    threshold = 0.2
    
    hash_coeffs = get_hash_coeffs(br)
    
    M = min_hashing(shingles, hash_coeffs, br)
    
    bucket_list = LSH(M, b, r, band_hash_size)
    
    predicted_ratings = calculate_ratings(
        movies, ratings_train, ratings_test, threshold, bucket_list, M, b, r, band_hash_size, shingles, movie_index_dict)
    
    ratings_test['rating'] = predicted_ratings

    ratings_test.to_csv(output_file, index=False)

    time_end = time.time()
    print("Time elapsed : ", (time_end-time_start)/60)
