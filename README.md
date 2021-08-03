# Myanimelist User Based Recommender: Project Overview
- Created a tool/app for the users of the online forum myanimelist. Based on collaborative filtering/hybrid methods the app recommends new anime for users based on implicit/explicit feedback from their animelist. App is hosted on https://myanimelistrecommender.herokuapp.com/
- Scraped anime list for over 4000 users using an edited version of the package "malscraper" and my own beautiful soup code. 
- Cleaned/pre-processed data from web-scraping using pivot tables, string manipulation, filters, masks, etc. 
- Created visualizations of collected data using plots from seaborn.
- Implemented LightFM function to create new recommendations based on user's implicit feedback, and content information about the individual anime. 
- Used Flask to build a barebones HTML website for webhosting. 
## Code and Resources Used
- Python Version 3.7
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, scipy, malscraper, mal, lightFM, Flask.
- For Web Framework Requirements: pip install -r requirements.txt
- malscraper: https://github.com/QasimK/mal-scraper

### [Web Scraping](https://github.com/AdamSabol89/MAL-recommender-user/blob/master/MAL_Recommender_Scraping.ipynb)
- Created beautiful soup code to scrape https://myanimelist.net/users.php for a list of users. 
- Edited the malscraper package to include information on: anime episodes, media type, airing status, studios, mpaa rating, and user_id. 
- Defined a function which scraped information on 4000 users and sent the scraped data to a pandas dataframe. 

### Data Cleaning
- Created two dataframes from scraped data, one based on user information, the second based on anime entry information.
- Filtered entries to only include those of media type "Special", "TV", and "Movie." 
- Removed duplicate user entries from scraping process. 
- Cleaned string information for anime studio entries. 
- Grouped low count studios and low count anime-num-episodes into "other" category.

### [Exploratory Data Analysis](https://github.com/AdamSabol89/MAL-recommender-user/blob/master/MAL_Recommender_Data_exploration.ipynb)
