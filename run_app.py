from flask import Flask, request
from flask import render_template
import pandas as pd
import numpy as np

app = Flask(__name__)


def weighted_rating(r, m=1838.4000000000015, C=6.092171559442016):
    v = r['vote_count']
    R = r['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


def demographic_Data(lang,ctry):
    fdata = pd.read_pickle('demographic.pkl')
    df = fdata[(fdata['spoken_languages'].apply(lambda x: lang in x)) & (fdata['production_countries'].apply(lambda x: ctry in x))]
    df['scores'] = df.apply(weighted_rating, axis=1)
    df = df.sort_values('scores', ascending=False)
    df = df[['title', 'vote_count', 'vote_average', 'scores']].head(5)
    df = df.reset_index(drop=True)
    return df


# Load the model
preds_df = pd.read_pickle("SVD_pickle.pkl")
ratings_df = pd.read_csv('ratings.csv' )
ratings_df.columns = ["UserID","MovieID","Rating","Timestamp"]
movies_df = pd.read_csv('movie.csv')
movies_df.columns = ["MovieID","Title","Genres"]
users = ratings_df['UserID'].unique().tolist()
merged_df = movies_df.merge(ratings_df, on='MovieID')
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

def recommend_movies_RBM(userID, num_recommendations):
    already_rated = merged_df[merged_df['UserID'] == userID]

    scored_movies_df_50 = pd.read_pickle('RBM_pickled.pkl')
    merged_df_50 = scored_movies_df_50.merge(already_rated, on='MovieID', how='outer')
    merged_df_50 = merged_df_50.drop('UserID', axis=1)
    recommendations = merged_df_50.sort_values(['Recommendation Score'], ascending=False).head(num_recommendations)
    recommendations = recommendations.loc[:, ['Title', 'Genres', 'Recommendation Score', 'Rating']]
    return already_rated, recommendations


def recommend_movies(userID, num_recommendations=5):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = ratings_df[ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').
                 sort_values(['Rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='MovieID',
                                 right_on='MovieID').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )
    recommendations = recommendations.loc[:, ['Title', 'Genres']]
    return user_full, recommendations


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/demo')
def demo():
    return render_template('demographic1.html')


@app.route('/recommend', methods=['POST'])
def recommend_movie():
    if request.method == 'POST':
        name = request.form['fname']
        con = request.form['country']
        lan = request.form['lang']
        dfr = demographic_Data(lan, con)
        data = {'username': name}
        return render_template('demographic2.html', data=data,
                               tables=[dfr.to_html(classes='dfr')], titles=['na', 'Trending Movies for you'])
    else:
        return 'Error'


@app.route('/rbm')
def rbm():
    return render_template('rbm1.html', users=users)


@app.route('/RBM_reccomendations', methods=['POST', 'GET'])
def result_RBM():
    u_id = request.form['user']
    already_rated, predictions = recommend_movies_RBM(int(u_id), 5)
    return render_template('rbm2.html', tables=[predictions.to_html(classes='predictions')],
    titles=['na', 'Top picks for you'])


@app.route('/collab')
def collab():
    return render_template('collaborative1.html', users=users)


@app.route('/recommendations',methods=['POST', 'GET'])
def result():
    u_id = request.form['user']
    already_rated, predictions = recommend_movies(int(u_id), 5)
    return render_template('collaborative2.html', tables=[predictions.to_html(classes='predictions')],
    titles=['na', 'Movies you may like'])


final_df=pd.read_csv('final_df.csv')
final_df = final_df.reset_index()
indices = pd.Series(final_df.index, index=final_df['title'])
cosine_sim=np.load('cosine.npy')
movies = list(pd.read_csv('movies.csv',header=None)[0])
movies.sort()
choices_mov = list(pd.read_csv('movies.csv',header=None)[0])
choices_mov.sort()


def get_recommendations(title, cosine_sim,indices,df):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].values


@app.route('/content')
def content():
    return render_template('content1.html', users=choices_mov)


@app.route('/next', methods=['POST'])
def next():
    if request.method == 'POST':
        req = request.form['movie']
        recc = get_recommendations(req, cosine_sim, indices,final_df)
        recc = pd.DataFrame(data=recc[0:],
                            columns=['movies'],
                            index=[x for x in range(len(recc))])
        return render_template("content2.html", movie=req, tables=[recc.to_html(classes='recc')],
                               titles=['na','Because you watched certain movies'])


if __name__ == '__main__':
    app.run(debug=True)
