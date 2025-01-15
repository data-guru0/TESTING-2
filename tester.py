from utils.helpers import *
import pandas as pd
from pipeline.prediction_pipeline import hybrid_recommendation
DF_PATH = "artifacts/processed/anime_df.csv"
DF_SYNOPSIS = "artifacts/processed/synopsis_df.csv"

ANIME_WEIGHTS = "artifacts/weights/anime_weights.pkl"
ANIME2ANIME_EN = "artifacts/processed/anime2anime_encoded.pkl"
ANIME2ANIME_DEC = "artifacts/processed/anime2anime_decoded.pkl"


USER_WEIGHTS = "artifacts/weights/user_weights.pkl"
USER2USER_EN = "artifacts/processed/user2user_encoded.pkl"
USER2USER_DEC = "artifacts/processed/user2user_decoded.pkl"

RATING_DF = "artifacts/processed/rating_df.csv"



#print(getAnimeFrame("Naruto",DF_PATH))

#print(getSypnopsis(1 , DF_SYNOPSIS))

# content_recommended_animes =find_similar_animes('Dragon Ball Z',ANIME_WEIGHTS,ANIME2ANIME_EN,ANIME2ANIME_DEC, DF_PATH,DF_SYNOPSIS, n=5, neg=False)
# print(content_recommended_animes.columns)

# # Check if the function works correctly
# similar_users = find_similar_users(6527, USER_WEIGHTS, USER2USER_EN, USER2USER_DEC, n=5, neg=False)
# user_pref = get_user_preferences(6527, RATING_DF, DF_PATH)
# user_recommended_animes = get_top_recommended_animes(similar_users, user_pref, DF_PATH, DF_SYNOPSIS, RATING_DF,n=10)
# print(user_recommended_animes.columns)


# content_recommended_animes_list = content_recommended_animes['name'].tolist()
# print(content_recommended_animes_list)
# user_recommended_animes_list = user_recommended_animes['anime_name'].tolist()  
# print(user_recommended_animes_list)


# hybrid_recs = hybrid_recommendation(content_recommended_animes_list, user_recommended_animes_list, content_weight=0.4, user_weight=0.6)
# print("Hybrid Recommendations:", hybrid_recs)


user_id = 6527
hybrid_recommendations = hybrid_recommendation(user_id, user_weight=0.6, content_weight=0.4)
print("Hybrid Recommendations:", hybrid_recommendations)

