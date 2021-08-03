import pandas as pd
import numpy as np
import string 
import seaborn as sns
import matplotlib.pyplot as plt
#Load in the data 
mal_raw_data = pd.read_csv("C:/Users/Localadmin/MAL Recommender/AnimeData")


#Check consumption status
mal_raw_data["consumption_status"].value_counts()

#Remove backlogs
#mal_consuming = mal_raw_data[mal_raw_data["consumption_status"]!="ConsumptionStatus.backlog"]
mal_consuming = mal_raw_data
mal_consuming = mal_consuming[mal_consuming['anime_media_type_string']!="Music"]
mal_consuming = mal_consuming[mal_consuming['anime_media_type_string']!="OVA"]
mal_consuming = mal_consuming[mal_consuming['anime_media_type_string']!="ONA"]

mal_consuming["anime_media_type_string"].unique()
mal_consuming['anime_media_type_string'].unique()
#Check distribution of scores
mal_consuming["score"].value_counts()
mal_consuming.drop_duplicates(inplace=True)

#Get user data
user_data_df = mal_consuming[["name", "score", "user_id", "consumption_status"]]
user_data_df["consumption_status"] = user_data_df["consumption_status"].str.split(".", expand=True)[1]

user_data_df["consumption_status"].value_counts()
#remove duplicates from scraping process
user_data_df.drop_duplicates(inplace=True)

#Get anime features
feature_data  = mal_consuming[['name','anime_num_episodes','anime_airing_status','anime_studios','anime_media_type_string','anime_mpaa_rating_string']]
feature_data.drop_duplicates(subset = "name", inplace=True)

#Text parsing/cleaning
feature_data['anime_studios'] = feature_data['anime_studios'].str.strip("[]{}")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("]", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("[", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("}", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace(":", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("id", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("name", "")
#duplicates['anime_studios'] = duplicates['anime_studios'].str.replace("\d+", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("\'", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace("\"", "")
feature_data['anime_studios'] = feature_data['anime_studios'].str.replace(",", "")
#seperate multiple studios
f = feature_data['anime_studios'].str.split(pat = '{', expand=True)

#clean dataframe
for col in f.columns: 
    f[col] = f[col].str.strip()
for col in f.columns: 
    f[col] = f[col].str.lstrip(string.digits)
for col in f.columns: 
    f[col] = f[col].str.strip()

f = f.replace(r'^\s*$', np.nan, regex=True)
f = f.fillna(value=np.nan)

#group low count studios into other
for i in range(len(f.columns)):
    col_values = f[i].value_counts()
    col_values = col_values.to_frame().reset_index()
    if i == 0:
        values = pd.DataFrame(col_values)
    else: 
        values = pd.merge(values, col_values, how="outer", on =["index"])
#add other column
values = values.fillna(0)
values["count"] = values.sum(axis = 1)
values.drop([0,1,2,3],axis = 1, inplace= True)
values["studio"] = ""
values.loc[values["count"] < 3, "studio"] = "other"

#replace values with other
for i, value in enumerate(values["studio"]):
    if value == "":
        values["studio"][i] = values["index"][i]
studios = values.studio.unique()
values.columns = ["key", "drop", "value"]
values = values.drop(["drop"], axis = 1)
dict_studios = dict(zip(values.key, values.value))

for col in f.columns: 
    f= f.replace({col:dict_studios})
#rename columns
f.columns = ['anime_studio1', 'anime_studio2','anime_studio3','anime_studio4']

#add data to original dataframe
feature_data = pd.concat([feature_data, f], axis= 1)

#grop low number of episodes into other
values = feature_data["anime_num_episodes"].astype("string").value_counts()
values = values.to_frame().reset_index()
values["episodes"] = ""
for i, value in enumerate(values["anime_num_episodes"]): 
    if value < 10:
        values["episodes"][i] = "other"
    else:
        values["episodes"][i] = values["index"][i]


values.columns = ["key", "drop", "value"]
values = values.drop(["drop"], axis = 1)
dict_episodes = dict(zip(values.key, values.value))
feature_data["anime_num_episodes"] = feature_data["anime_num_episodes"].astype("string")
feature_data = feature_data.replace({"anime_num_episodes": dict_episodes})
feature_data["anime_airing_status"] = feature_data["anime_airing_status"].astype("string")
item_data_df = feature_data.drop(["anime_studios"], axis = 1)


#save dataframes
user_data_df.to_csv("C:/Users/Localadmin/MAL Recommender/user_data", header=True, index= False)
item_data_df.to_csv("C:/Users/Localadmin/MAL Recommender/item_data", header=True, index= False)

