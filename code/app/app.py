from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


def get_quality_label(track_quality):
    if track_quality== 1:
        label = 'High Quality'
    elif track_quality == 0.7:
        label = 'Above Average'
    elif track_quality == 0.5:
        label = 'Average Quality'
    else:
        label = 'Below Average'
        
    return label


# 1:  Top Chart (Interest factor > 0.09) x =1 - more than 290,000 interests
# 0.8: between 230,000 and 290,000 
# 0.7: between 165,000 and 230,000 
# 0.6: between 98,000 and 165,000 
# 0.5: between 65,000 and 98,000 
# 0.3: between 30,000 and 65,000 
# 0.2: below 30,000 would be low interest 

def get_Rank_label(prediction):
    if prediction > 0.8:
        label = 'Top Chart (more than 290,000 favorites)'
    elif 0.7 < prediction <= 0.8:
        label = 'High Rank (between 230,000 and 290,000 favorites)'    
    elif 0.6 < prediction <= 0.7:
        label = 'Above Average Rank (between 165,000 and 230,000 favorites)'
    elif 0.5 < prediction <= 0.6:
        label = 'Average Rank (between 98,000 and 165,000 favorites)'
    elif 0.3 < prediction <= 0.5:
        label = 'Below Average Rank (between 65,000 and 165,0000 favorites)'
    elif 0.2 < prediction <= 0.3:
        label = 'Below Average Rank (between 30,000 and 65,0000 favorites)'
    else:
        label = 'Low Rank (below 30,000 favorites)'
        
    return label


@app.route('/predict_price', methods=['POST', 'GET'])
def predict_price():
    # get the parameters
    listens = float(request.form['listens'])
    track_year = float(request.form['track_year'])
    track_quality = float(request.form['track_quality'])

    # load the model and predict
    model = joblib.load('model/RandomForest10-model.pkl')
    prediction = model.predict([[listens, track_year, track_quality]])
    print("RANK: ", prediction)
    track_rank = get_Rank_label(pd.to_numeric(prediction))
    track_quality_label = get_quality_label(track_quality)

    return render_template('results.html',
                           listens=int(listens),
                           track_year=int(track_year),
                           track_quality= track_quality_label,
                           track_rank=track_rank
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
