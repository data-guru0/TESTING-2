import pandas as pd
import numpy as np
import joblib
from config.paths_config import *
########################## ANIME FRAME #################################33

def getAnimeFrame(anime , path_df):
    df = pd.read_csv(path_df)
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        return df[df.eng_version == anime]
    
#############################  ANIME SYNPOSIS #############################333333

def getSypnopsis(anime,path_sypnopsis_df):
    sypnopsis_df = pd.read_csv(path_sypnopsis_df)
    if isinstance(anime, int):
        return sypnopsis_df[sypnopsis_df.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return sypnopsis_df[sypnopsis_df.Name == anime].sypnopsis.values[0]
    

####################################    CONTENT BASED RECOMMNDATION ###########################################

def find_similar_animes(name, path_anime_weights , path_anime2anime_encoded , path_anime2anime_decoded, path_df , synopsis_df , n=10, return_dist=False, neg=False):
    try:

        anime2anime_encoded = joblib.load(path_anime2anime_encoded)
        anime_weights = joblib.load(path_anime_weights)
        anime2anime_decoded = joblib.load(path_anime2anime_decoded)

        index = getAnimeFrame(name,path_df).anime_id.values[0]
        encoded_index = anime2anime_encoded.get(index)

        weights = anime_weights
        
        
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
       
        n = n + 1            
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        print('Animes closest to {}'.format(name))


        
        if return_dist:
            return dists, closest
        
        
        SimilarityArr = []

        for close in closest:

            
            decoded_id = anime2anime_decoded.get(close)

            
            sypnopsis = getSypnopsis(decoded_id,synopsis_df)

            
            anime_frame = getAnimeFrame(decoded_id,path_df)
            
            
            anime_name = anime_frame.eng_version.values[0]
            genre = anime_frame.Genres.values[0]
            similarity = dists[close]
            SimilarityArr.append({"anime_id": decoded_id, "name": anime_name,
                                  "similarity": similarity,"genre": genre,
                                  'sypnopsis': sypnopsis})

        Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
        return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)

    except:
        print('{}!, Not Found in Anime list'.format(name))

################################# USER-BASED-RECOMMEND ##############################3333




def find_similar_users(item_input, path_user_weights , path_user_encoded , path_user_decoded, n=10,return_dist=False, neg=False):
    try:
        user2user_encoded = joblib.load(path_user_encoded)
        user2user_decoded = joblib.load(path_user_decoded)
        user_weights = joblib.load(path_user_weights)


        index = item_input
        encoded_index = user2user_encoded.get(index)
        weights = user_weights
    
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        
        n = n + 1
        
        if neg:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        print('> users similar to #{}'.format(item_input))

        if return_dist:
            return dists, closest
        
        SimilarityArr = []
        
        for close in closest:
            similarity = dists[close]

            if isinstance(item_input, int):
                decoded_id = user2user_decoded.get(close)
                SimilarityArr.append({"similar_users": decoded_id, 
                                      "similarity": similarity})

        similar_users = pd.DataFrame(SimilarityArr).sort_values(by="similarity", 
                                                        ascending=False)
        similar_users = similar_users[similar_users.similar_users != item_input]
        
        return similar_users
    
    except:
        print('{}!, Not Found in User list'.format(item_input))


def get_user_preferences(user_id, path_rating_df , path_df , verbose=0):

    rating_df = pd.read_csv(path_rating_df)
    df = pd.read_csv(path_df)

    ## retrieves all anime that the user has rated.
    animes_watched_by_user = rating_df[rating_df.user_id==user_id]

    ## This means it finds the rating value that is higher than 75% of the ratings given by the user.
    ## Basically it is finding top ratings by user...
    user_rating_percentile = np.percentile(animes_watched_by_user.rating, 75)

    ## Filters out all ratings below the 75th percentile, keeping only the user's highest-rated anime.
    animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]

    ## Sorts the user's top-rated anime in descending order of ratings.
    top_animes_user = (
        animes_watched_by_user.sort_values(by="rating", ascending=False)#.head(10)
        .anime_id.values
    )
    
    ## Extract only those highly rated anime from main dataframe
    anime_df_rows = df[df["anime_id"].isin(top_animes_user)]
    anime_df_rows = anime_df_rows[["eng_version", "Genres"]]
    
    if verbose != 0:
        print("> User #{} has rated {} movies (avg. rating = {:.1f})".format(
          user_id, len(animes_watched_by_user),
          animes_watched_by_user['rating'].mean(),
        ))
        
    return anime_df_rows


def get_top_recommended_animes(similar_users, user_pref, path_df, synopsis_df, path_rating_df, n=10):
    recommended_animes = []
    anime_list = []

    ## Loop through Similar Users
    for user_id in similar_users.similar_users.values:
        pref_list = get_user_preferences(int(user_id), path_rating_df, path_df, verbose=0)

        # Filter out animes already rated by the current user
        pref_list = pref_list[~pref_list.eng_version.isin(user_pref.eng_version.values)]

        if not pref_list.empty:
            anime_list.append(pref_list.eng_version.values)

    if anime_list:
        anime_list = pd.DataFrame(anime_list)
        
        # Get the top n recommended animes
        sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)

        for i, anime_name in enumerate(sorted_list.index):
            n_user_pref = sorted_list[sorted_list.index == anime_name].values[0][0]
            if isinstance(anime_name, str):
                try:
                    frame = getAnimeFrame(anime_name, path_df)
                    anime_id = frame.anime_id.values[0]
                    genre = frame.Genres.values[0]
                    sypnopsis = getSypnopsis(int(anime_id), synopsis_df)
                    recommended_animes.append({
                        "n": n_user_pref,
                        "anime_name": anime_name,
                        "Genres": genre,
                        "sypnopsis": sypnopsis
                    })
                except Exception as e:
                    print(f"Error fetching details for {anime_name}: {e}")

    return pd.DataFrame(recommended_animes).head(n)  # Only return the top n recommendations






