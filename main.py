import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify


data = pd.read_csv('final.csv')
data.drop(columns = ['id'], inplace = True)

data.set_index('Hotel Name',inplace=True)

count = CountVectorizer()
count_matrix = count.fit_transform(data['Reviews_Key_words'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(data.index)


def recommendations(title, cosine_sim=cosine_sim):
    # initializing the empty list of recommended movies
    recommended_hotels = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_hotels.append(list(data.index)[i])

    return recommended_hotels



print(recommendations('Abbazia_di_Novacella-Varna_South_Tyrol_Province_Trentino_Alto_Adige'))


###############



app = Flask(__name__)
@app.route('/api/', methods=['POST'])
def api():
     hotel = request.json['hotel']
     prediction = recommendations(hotel)
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    app.run()