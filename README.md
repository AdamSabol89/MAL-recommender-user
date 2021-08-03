# Myanimelist User Based Recommender: Project Overview
- Created a tool/app for the users of the online forum myanimelist. Based on collaborative filtering/hybrid methods the app recommends new anime for users based on implicit/explicit feedback from their animelist. App is hosted on https://myanimelistrecommender.herokuapp.com/
- Scraped anime list for over 4000 users using an edited version of the package "malscraper" and my own beautiful soup code. 
- Cleaned/pre-processed data from web-scraping using pivot tables, string manipulation, filters, masks, etc. 
- Created visualizations of collected data using plots from seaborn.
- Implemented LightFM function to create new recommendations based on user's implicit feedback, content information about the individual anime, and explicit user scores. 
- Created a pipeline to take in information from myanimelist site and output user predictions.
- Used Flask to build a basic HTML based website for model webhosting. 
## Code and Resources Used
- Python Version 3.7
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, scipy, malscraper, mal, lightFM, Flask.
- For Web Framework Requirements: pip install -r requirements.txt
- malscraper: https://github.com/QasimK/mal-scraper

### [Web Scraping](https://github.com/AdamSabol89/MAL-recommender-user/blob/master/MAL_Recommender_Scraping.ipynb)
- Created beautiful soup code to scrape https://myanimelist.net/users.php for a list of users. 
- Edited the mal-scraper package to include information on: anime episodes, media type, airing status, studios, mpaa rating, and user_id. 
- Defined a function which scraped information on 4000 users and sent the scraped data to a pandas dataframe. 

### Data Cleaning
- Created two dataframes from scraped data, one based on user information, the second based on anime entry information.
- Filtered entries to only include those of media type "Special", "TV", and "Movie." 
- Removed duplicate user entries from scraping process. 
- Cleaned string information for anime studio column. 
- Grouped low count studios and low count anime-num-episodes into "other" category.

### [Exploratory Data Analysis](https://github.com/AdamSabol89/MAL-recommender-user/blob/master/MAL_Recommender_Data_exploration.ipynb)
- Did basic exploratory analysis on user information such as scores/consumption_status. 
- Full results are stated in the notebook linked above, provided below are some examples from the analysis. 
<p align="center">
  <img src="https://github.com/AdamSabol89/MAL-recommender-user/blob/master/figures/mean_count_plot_scores.png">
</p>
<p align="center">
  <img src="https://github.com/AdamSabol89/MAL-recommender-user/blob/master/figures/mean_user_score.png">
</p>
<p align="center">
  <img src="https://github.com/AdamSabol89/MAL-recommender-user/blob/master/figures/ratio_dropped.png">
</p>

### Data Pre-Processing for Model Building 
- LightFM, the package used for the modeling in this project, takes scipy-sparse matrixes as inputs. Some work must be done to get our data from its current format into this format.
- Removed anime with <25 entries in the dataset for better model performance. 
- Pivoted user data to create implicit information for interactions matrix. 
- Created one-hot-encoding dataframe for item data.
- Standardized user scores and reintroduced them as user-features. 
- Sent all dataframes to scipy sparse matrices and saved list of anime titles. 

### [Model Building](https://github.com/AdamSabol89/MAL-recommender-user/blob/master/MAL_Recommender_Modeling_Notebook.ipynb) 

