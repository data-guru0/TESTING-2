from config.paths_config import *
from utils.helpers import *


def hybrid_recommendation(user_id, user_weight=0.6, content_weight=0.4):
    # Step 1: User-Based Recommendation
    similar_users = find_similar_users(user_id, USER_WEIGHTS, USER2USER_ENCODED, USER2USER_DECODED, n=5, neg=False)
    user_pref = get_user_preferences(user_id, RATING_DF, DF_PATH)
    user_recommended_animes = get_top_recommended_animes(similar_users, user_pref, DF_PATH, DF_SYNOPSIS, RATING_DF, n=10)
    
    # Extract user-recommended anime names
    user_recommended_animes_list = user_recommended_animes['anime_name'].tolist()
    
    # Step 2: Content-Based Recommendation
    content_recommended_animes = []
    for anime in user_recommended_animes_list:
        similar_animes = find_similar_animes(anime, ANIME_WEIGHTS, ANIME2ANIME_ENCODED, ANIME2ANIME_DECODED, DF_PATH, DF_SYNOPSIS, n=5, neg=False)
        if similar_animes is not None and not similar_animes.empty:  # Check if the result is valid
            content_recommended_animes.extend(similar_animes['name'].tolist())  # Adjust column name as needed
        else:
            print(f"No similar animes found for: {anime}")
    
    # Step 3: Combine Recommendations with Weights
    combined_scores = {}
    
    # Add user-based recommendations with weights
    for anime in user_recommended_animes_list:
        combined_scores[anime] = combined_scores.get(anime, 0) + user_weight
    
    # Add content-based recommendations with weights
    for anime in content_recommended_animes:
        combined_scores[anime] = combined_scores.get(anime, 0) + content_weight
    
    # Sort by combined score in descending order
    sorted_animes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return [anime for anime, score in sorted_animes[:10]]