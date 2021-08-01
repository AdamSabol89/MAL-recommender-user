import pandas as pd
import numpy as np 
import scipy 
from sklearn.preprocessing import StandardScaler
import pickle 

#load userdata
user_data = pd.read_csv("user_data")
#Build user interactions based off of implicit data
user_data["implicit"] = 1
user_interactions = user_data.pivot_table(index = ["user_id"], columns = ["name"], values = "implicit")

#Remove low count anime 
for col in user_interactions.columns: 
    count_anime = user_interactions[col].count()
    if count_anime < 25: 
        user_interactions = user_interactions.drop(col, axis=1)
#add 0's for sparse matrix
user_interactions.fillna(0, inplace = True)
#save as scipy sparse matrix
user_interactions = scipy.sparse.csr_matrix(user_interactions.values)
scipy.sparse.save_npz("user_interactions", user_interactions)

#save anime listed in user interactions matrix for later use
keep_anime = user_interactions.columns
anime_list = list(keep_anime)
with open ("anime_list.txt", "wb") as fp: 
    pickle.dump(anime_list, fp)  
    
#Build item features 
item_data = pd.read_csv("item_data")

#extract anime studios for mult-encoding 
studio_data = item_data[["anime_studio1", "anime_studio2", "anime_studio3", "anime_studio4"]]

#get list of all unique studios
anime_studios = set(studio_data["anime_studio1"].dropna().unique())
anime_studios = set(studio_data["anime_studio2"].dropna().unique()).union(anime_studios)
anime_studios = set(studio_data["anime_studio3"].dropna().unique()).union(anime_studios)
anime_studios = set(studio_data["anime_studio4"].dropna().unique()).union(anime_studios)

anime_studios = list(anime_studios)

#create empty dataframe of studios
studio_df = pd.DataFrame(columns = anime_studios)

#merge studio dataframes
for col in studio_data.columns :
    temp = pd.get_dummies(studio_data[col])
    temp = studio_df.append(temp)
    if col == "anime_studio1":
        values = temp.values
    else: 
        values = values + temp.values

studio_dummies = pd.DataFrame(data=values, columns = anime_studios).fillna(0)

#replace values from addition 

#get dummies for regular attributes
item_data["anime_airing_status"] = item_data["anime_airing_status"].astype(str)

item_data_dummies = pd.get_dummies(item_data[["anime_num_episodes", "anime_airing_status","anime_media_type_string","anime_mpaa_rating_string"]])

#reset index and concat dataframes
item_data_dummies.index = item_data["name"]
studio_dummies.index = item_data["name"]
item_data_dummies = pd.concat([item_data_dummies, studio_dummies], axis= 1)

#filter so only anime in userinteractions are retained
item_data_dummies = item_data_dummies.filter(items = anime_list, axis = 0)

#save as scipy sparse matrix
item_features = scipy.sparse.csr_matrix(item_data_dummies.values)
scipy.sparse.save_npz("item_features", item_features)


#Build user features based on user ratings
user_features = user_data 
#get only rated anime
user_features = user_features.loc[user_features["score"] != 0]
user_features = user_features.pivot_table(index = ["user_id"], columns = ["name"], values = "score")
#add back in all users
users= pd.DataFrame(data = None, index = user_interactions.index)
user_features = users.merge(user_features, how = "outer", right_index = True, left_index = True)
user_features = user_features.filter(items = anime_list, axis = 1)

#add back in all anime
full_anime = pd.DataFrame(data=None, columns = user_interactions.columns)
user_features = full_anime.append(user_features)

#standardize ratings by user
scaler = StandardScaler()
user_features_transpose = user_features.transpose()
scaler.fit(user_features_transpose)
user_features_transpose = scaler.transform(user_features_transpose)
user_features = pd.DataFrame(data = user_features_transpose.transpose()).fillna(0)
user_features = scipy.sparse.csr_matrix(user_features.values)
scipy.sparse.save_npz("user_features", user_features)


#need to remove twos from item features
