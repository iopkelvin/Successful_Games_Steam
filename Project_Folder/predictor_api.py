"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np

#open the pickled XGBoost Classifier
with open("lr.pickle", "rb") as f:
    xgb_model = pickle.load(f)

#be careful about the feature names, xgb_classifier does not properly save feature names
#so I had to rename them as they were saved in the pickle i.e. ['f0', 'f1',... ,'f15']

feature_names = {'f' + str(i) for i in range(15)}

#original column names for my model
column_names = ['singleplayer', 'steam cloud', 'indie', 'anime', 'multiplayer', 'very_cheap', 'steam trading cards', 'great_soundtrack', 'classic', 'vr', 'comedy', 'mac', 'puzzle', 'cheap', 'co_op']


def make_api_prediction(feature_dict):
    """
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns a dictionary with the following keys
      all_probs: a list of dictionaries with keys 'name', 'prob'. This tells the
                 probability of class 'name' appearing is the value in 'prob'
      most_likely_class_name: string (name of the most likely class)
      most_likely_class_prob: float (name of the most likely probability)
    """

    #be careful about the feature names, xgb_classifier does not properly save feature names
    #so I had to rename them as they were saved in the pickle i.e. ['f0', 'f1',... ,'f15']

    import pprint as pp

    pp.pprint(feature_dict) #sanity check, outputs to terminal running routes.py

    #this is just me working with my data
    title_length = int(feature_dict['singleplayer'])
    goal = round(float(feature_dict['steam_cloud']), 2)
    main_country = feature_dict['indie']
    main_country = feature_dict['anime']
    backer_cnt = feature_dict['multiplayer']
    project_duration = float(feature_dict['very_cheap'])

    # {'singleplayer':singleplayer,
    #                 'steam_cloud': steam_cloud,
    #                 'indie': indie,
    #                 'anime': anime,
    #                 'multiplayer': multiplayer,
    #                 'very_cheap':very_cheap}

    f_dict = {'f' + str(i):0 for i in range(100)}

    f_dict['f0'] = title_length
    f_dict['f1'] = goal

    if backer_cnt == 'yes':
        f_dict['f2'] = 1

    if main_country == 'US':
        f_dict['f3'] = 1

    f_dict['f4'] = float(round(project_duration/60, 2))

    categories = {'f' + str(i):column_names[i] for i in range(5, 16)}

    for key, value in categories.items():
        if main_country == value:
            f_dict[key] = 1


    x_input = [f_dict[name] for name in feature_names]
    x_input = [0 if val == '' else float(val) for val in x_input] #if there are no values, give 0

    pred_probs = xgb_model.predict_proba([x_input]).flat

    names = ['failure', 'success'] #replace with names of your target variable
                                   #make sure to check order against some test predictions

    probs = [{'name': names[index], 'prob': float(pred_probs[index])}
              for index in np.argsort(pred_probs)[::-1]]

    #response is a dictionary to return
    response = {
        'all_probs': probs,
        'most_likely_class_name': probs[0]['name'],
        'most_likely_class_prob': float(probs[0]['prob']),
    }

    return response

# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what setting all params to 0 predicts")
    features = {f:'0' for f in feature_names}
    print('Features are')
    pprint(features)

    response = make_api_prediction(features)
    print("The returned object")
    pprint(response)
