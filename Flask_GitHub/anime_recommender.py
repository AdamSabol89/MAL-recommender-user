import pandas as pd 
import numpy as np 
import pickle 
import scipy 
from lightfm import lightfm
import my_malscraper
import mal
from sklearn.preprocessing import StandardScaler

def get_user(new_user):
    new_user_anime = my_malscraper.get_user_anime_list(new_user)
    new_user_anime = pd.DataFrame(new_user_anime)    
    return(new_user_anime)

def pre_processer(new_user_df):
    new_user_df = new_user_df[new_user_df['anime_media_type_string']!="Music"]
    new_user_df = new_user_df[new_user_df['anime_media_type_string']!="OVA"]
    new_user_df = new_user_df[new_user_df['anime_media_type_string']!="ONA"]
    new_user_df = new_user_df[["name", "score", "user_id"]]
    with open("anime_list.txt", "rb") as fp:   
        anime_list  = pickle.load(fp)
    new_user_df.index = new_user_df["name"]
    new_user_df = new_user_df.filter(anime_list, axis = 0)
    new_user_df.reset_index(inplace=True, drop = True)
    return(new_user_df)


def get_user_interactions(new_user_df):
    #with open("user_features.txt", "rb") as fp:
     #   user_features = pickle.load(fp)
    with open("anime_list.txt", "rb") as fp:   
        user_interactions = pickle.load(fp)
    new_user_df["implicit"] =1 
    
    user_interactions = pd.DataFrame(columns = user_interactions)
    
    pivoted = new_user_df.pivot_table(index = "user_id", columns = "name", values = "implicit")
    
    user_interactions = user_interactions.append(pivoted)
    user_interactions.fillna(0, inplace= True)
    
    new_user_interactions = scipy.sparse.csr_matrix(user_interactions.values)
    
    return(new_user_interactions)

def get_user_features(new_user_df):
    with open("anime_list.txt", "rb") as fp:   
        user_features = pickle.load(fp)
        
    user_features = pd.DataFrame(columns = user_features)
    
    new_user_df = new_user_df[new_user_df["score"]!= 0]
    pivoted = new_user_df.pivot_table(index = "user_id", columns = "name", values = "score")
    
    pivoted_transpose = np.array(pivoted).transpose()
    scaler = StandardScaler()
    scaler.fit(pivoted_transpose)
    pivoted_transpose = scaler.transform(pivoted_transpose)
    
    pivoted = pd.DataFrame(data = pivoted_transpose.transpose(), columns = pivoted.columns)
    
    user_features = user_features.append(pivoted)
    user_features.fillna(0,inplace = True)
    
    new_user_features = scipy.sparse.csr_matrix(user_features.values)
    
    return(new_user_features)

def get_model_matrices(new_user_df):
    new_user_interactions = get_user_interactions(new_user_df)
    new_user_features = get_user_features(new_user_df)
    
    user_interactions = scipy.sparse.load_npz("user_interactions.npz")
    user_features = scipy.sparse.load_npz("user_features.npz")
    item_features = scipy.sparse.load_npz("item_features.npz")
    
    new_user_interactions = scipy.sparse.vstack([user_interactions, new_user_interactions])
    new_user_features = scipy.sparse.vstack([user_features, new_user_features])
    
    user_shape = new_user_interactions.shape[0]
    feature_shape = item_features.shape[0]
    
    user_identity = np.identity(user_shape)
    user_identity = scipy.sparse.csr_matrix(user_identity)
    
    feature_identity = np.identity(feature_shape)
    feature_identity = scipy.sparse.csr_matrix(feature_identity)
    
    final_user_interactions = new_user_interactions
    final_user_features = scipy.sparse.hstack([new_user_features, user_identity])
    final_item_features = scipy.sparse.hstack([item_features, feature_identity])
    
    return final_user_interactions, final_user_features, final_item_features

def get_predictions(new_user_df, hybrid, loss):
    user_interactions, user_features, item_features = get_model_matrices(new_user_df)
    model = lightfm.LightFM(loss = loss)

    if hybrid == True: 
        model.fit(interactions=user_interactions, item_features = item_features, user_features =user_features)
        user = user_interactions.shape[0]-1
        user_row = user_interactions.getrow(user)
        user_array = user_row.toarray()
        user_for_pred = np.ndarray(shape = (3988), dtype=int, order='F')
        user_seen = np.argwhere(user_array != 0)
        user_for_pred.fill(user)
        items = np.arange(0,3988)
        predictions = np.array(model.predict(user_ids = user_for_pred, item_ids = items, item_features = item_features))
        new_recs = np.delete(predictions, user_seen)
        new_recs = new_recs.argsort()[-10:][::-1]
    else:

        model.fit(interactions=user_interactions)

        user = user_interactions.shape[0]-1
        user_row = user_interactions.getrow(user)
        user_array = user_row.toarray()
        
        user_seen = np.argwhere(user_array != 0)
        
        user_for_pred = np.ndarray(shape = (3988), dtype=int, order='F')
        user_for_pred.fill(user)
        
        items = np.arange(0,3988)
        predictions = np.array(model.predict(user_ids = user_for_pred, item_ids = items))
    
        new_recs = np.delete(predictions, user_seen)
        new_recs = new_recs.argsort()[-10:][::-1]
    
    return(new_recs)   

def web_recommender(username):
    user = get_user(username)
    user = pre_processer(user)
    predictions = get_predictions(user, hybrid= False, loss= "bpr")
    with open("anime_list.txt", "rb") as fp:   
        anime_list = pickle.load(fp)
    predicted_anime = []
    anime_urls = []
    for i in predictions: 
        predicted_anime.append(anime_list[i])
    for anime in predicted_anime:
        anime_url = mal.AnimeSearch(anime)
        anime_urls.append(anime_url.results[0].url)
    return predicted_anime, anime_urls

