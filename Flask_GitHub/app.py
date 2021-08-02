from flask import Flask, render_template, request
from anime_recommender import web_recommender

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def my_form_post():
    username = request.form['text']
    #processed_text = text.upper()
    anime_list, anime_urls = web_recommender(username)
    recommendations = zip(anime_list,anime_urls)
    return render_template("index.html", recommendations = recommendations)

if __name__ == '__main__':
    app.run(port = 3000, debug = True)