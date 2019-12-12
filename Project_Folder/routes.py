import flask
from flask import request
from predictor_api import make_api_prediction, feature_names #this is from predictor_api.py
from forms import Form #this is from forms.py
from collections import defaultdict
import os
# Initialize the app

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = '12345' #secret key is necessary, but it can be any value
# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!


@app.route("/", methods=["GET","POST"]) #just POST is enough to get info
def get_predict_page():
    "Returns the rendered page"
    form = Form()
    if form.validate_on_submit(): #this makes sure your form has data
        #here I am processing what form gave me, use .data to extract the data from the fields
        #see forms.py for more information about the fields
        singleplayer = form.singleplayer.data
        steam_cloud = form.steam_cloud.data
        indie = form.indie.data
        anime = form.anime.data
        multiplayer = form.multiplayer.data
        very_cheap = form.very_cheap.data

        feature_dict = {'singleplayer':singleplayer,
                        'steam_cloud': steam_cloud,
                        'indie': indie,
                        'anime': anime,
                        'multiplayer': multiplayer,
                        'very_cheap':very_cheap}

        # cols = ['singleplayer', 'steam_cloud', 'indie', 'anime', 'multiplayer', 'very_cheap', 'steam trading cards',
        # 'great_soundtrack', 'classic', 'vr', 'comedy', 'mac', 'puzzle', 'cheap', 'co_op']


        #I pass my processed form data to my model to make a prediction, see prediction_app.py
        predictions = make_api_prediction(feature_dict)

        probs = predictions['all_probs']
        most_likely_class_name = probs[0]['name']
        most_likely_class_prob = probs[0]['prob']
        least_likely_class_name = probs[1]['prob']
        least_likely_class_prob = probs[1]['prob']

        #I pass my form and other variables to render_template so that the html can receive values
        #to render on the page
        return flask.render_template('predictor.html',
                                    most_likely_class_name = most_likely_class_name,
                                    most_likely_class_prob = most_likely_class_prob,
                                    least_likely_class_name = least_likely_class_name,
                                    least_likely_class_prob = least_likely_class_prob,
                                    form = form)

    #if the form is invalid, no values are shown
    #this is the default
    return flask.render_template('predictor.html',
                                most_likely_class_name = "",
                                most_likely_class_prob = "",
                                least_likely_class_name = "",
                                least_likely_class_prob = "",
                                form = form)


# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4432)))

# For public web serving:
# app.run(host='0.0.0.0')
