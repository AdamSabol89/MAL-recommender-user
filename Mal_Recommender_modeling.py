import pandas as pd
import numpy as np
import scipy
import lightfm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

user_data = pd.read_csv("user_data")
item_data = pd.read_csv("item_data")

#Start with user item data
#Transform score into either positive or negative based on results from data exploration
user_data["rankings"] = 0 
user_data.loc[(user_data["score"] > 6), "rankings"] = 1

user_data.loc[(user_data["score"] == 0) & (user_data["consumption_status"] == "dropped"), "score"] = 5

user_data.loc[(user_data["rankings"] == 0), "rankings"] = -1

user_data.head(300)

#Experiment with normalization and mean filling 
user_data.loc[(user_data["score"] == 0) & (user_data["consumption_status"] == "dropped"), "score"] = 5
user_interactions = user_data.pivot_table(index = ["user_id"], columns = ["name"], values = "score")

holder = list(user_interactions.columns)
drop = list(set(list(temp.columns)) - set(holder))
temp = temp.drop(drop, axis=1)
user_interactions = user_interactions.drop(drop, axis = 1)


for col in user_interactions.columns: 
    count_anime = user_interactions[col].count()
    if count_anime < 25: 
        user_interactions = user_interactions.drop(col, axis=1)

user_data_filt = user_data[user_data["score"] != 0 ]
mean_anime_scores = user_data_filt.groupby("name").mean()
mean_anime_scores = {}

#merge and combine 
user_data = user_data.merge(mean_anime_scores, how = "right", on = "name")

user_data.mask()
    
#Problem with above ^ when calulating mean it is using 0 observations in calculation if 0 is only obs gives it as mean
#may need to filter dataframe such that anime with <n number of anime are ommitted. 
    
if 0 in user_interactions.values:
    print('Element exists in Dataframe')
    
    
#Standardized scalimg
scaler = StandardScaler()
user_interactions_transpose = user_interactions.transpose()
scaler.fit(user_interactions_transpose)
user_interactions_transpose = scaler.transform(user_interactions_transpose)

#Scale between 0 and 1 
scaler2 = MinMaxScaler((0,1))
scaler2.fit(user_interactions_transpose)
user_interactions_transpose = scaler2.transform(user_interactions_transpose)

#Filter based on top 30%
user_interactions_transpose = np.where(user_interactions_transpose >= .7, 1, user_interactions_transpose)
user_interactions_transpose = np.where(user_interactions_transpose < .7, -1, user_interactions_transpose)

#Return to user_interactions 
user_interactions_final = user_interactions_transpose.transpose()
user_interactions_final = pd.DataFrame(user_interactions_final)
user_interactions_final.columns = user_interactions.columns
user_interactions_final.index = user_interactions.index
user_interactions_final = user_interactions_final.fillna(0)
user_interactions = user_interactions_final
#standardize across user 




#Pivot the data 
user_interactions = user_data.pivot_table(index = ["user_id"], columns = ["name"], values = "ranking")

user_interactions = user_interactions.fillna(0)

#Now item feature data dummies
#item features regular dummies 
item_data["anime_airing_status"] = item_data["anime_airing_status"].astype("string")

item_feat_dummies = pd.get_dummies(item_data[["anime_num_episodes", "anime_media_type_string","anime_mpaa_rating_string"]])

airing_dummies = pd.get_dummies(item_data["anime_airing_status"])

item_feat_dummies = pd.concat([item_feat_dummies, airing_dummies], axis=1)

#Dummies for anime studios
ani_studios = ['anime_studio1', 'anime_studio2','anime_studio3','anime_studio4']
studio_dummies = pd.DataFrame(data= None, columns = studios)

for value in ani_studios:
    studio = pd.get_dummies(item_data[value])
    
    studio = studio_dummies.append(studio)
    
    studio = studio.fillna(0)
    
    if value == "anime_studio1":
        values = studio.values
        
    else: 
        values = values + studio.values

studio_dummies = pd.DataFrame(data = values, columns = studios)    

item_feat_dummies = pd.concat([item_feat_dummies, studio_dummies], axis = 1)

item_feat_dummies.index = item_data["name"]

temp = item_feat_dummies.transpose()

temp2 = temp.reindex(user_interactions.columns, axis=1)

temp2 = temp2.replace({2:1})
temp2 = temp2.transpose()
train_interactions = scipy.sparse.csr_matrix(user_interactions.values)
train_item_features = scipy.sparse.csr_matrix(temp2.values)

model = lightfm.LightFM(loss = 'warp')
model.fit(interactions=train_interactions, item_features =train_item_features)

train_interactions.shape
train_item_features.shape

train
train_item_features.shape[0]
##
user = np.ndarray(shape = (3221), dtype=int, order='F')
user.fill(1)
items = np.arange(0,3221)
arr = np.array(model.predict(user_ids = user, item_ids = items, item_features = train_item_features))
#arr.argsort()[-3:][::-1]
np.argmax(arr)


inverse_weights = []
for i in range(0,3221):
    weight = 8442/np.count_nonzero(user_interactions[user_interactions.columns[i]])
    inverse_weights.append(weight)  
inverse_weights =np.array(inverse_weights)

list2 = []
for i in range(len(user_interactions.columns)):
    t = np.count_nonzero(user_interactions[user_interactions.columns[i]])
    list2.append(t)
list2 = pd.Series(list2)
list2.value_counts()

anime = list(user_interactions.columns)

# Full user interaction matrix 
user_scores = np.array(user_interactions.iloc[1])
seen_anime = np.argwhere(user_scores != 0)
#weighted_preds= inverse_weights.T*arr
weighted_preds = arr 
new_recs = np.delete(weighted_preds, seen_anime)
recs = new_recs.argsort()[-10:][::-1]
for i in recs:
    print(anime[i])

user_interactions.index[4181]

len(items)
len(user)
#Testing user predictions
user1 = user_data[user_data["user_id"] == "BoopleDoople"]
user1["rankings"] = 0 
user1.loc[(user_data["score"] > 6), "rankings"] = 1
user1.loc[(user_data["score"] == 0) & (user_data["consumption_status"] != "dropped"), "rankings"] = 1
user1.loc[(user_data["rankings"] == 0), "rankings"] = -1
user_data.head(300)

user1_interactions = user1.pivot_table(index = ["user_id"], columns = ["name"], values = "rankings")
data = pd.DataFrame(data=None, columns = user_interactions.columns)

data =  data.append(user1_interactions).fillna(0)

#fix user feature matrix 