from flask import Blueprint, request, render_template
from ..load import processing_results, api
import string


twitter_mod = Blueprint('twitter', __name__, template_folder='templates', static_folder='static')

ascii_chars = set(string.printable)
ascii_chars.remove(' ')


def takeout_non_ascii(s):
    return list(filter(lambda x: x not in ascii_chars, s))


@twitter_mod.route('/twitter', methods=['GET', 'POST'])
def twitter():
    if request.method == 'POST':

        public_tweets = api.search(request.form['topic'], lang='hi', rpp=100)
        text = []
        for tweet in public_tweets:
            temp = ''.join(takeout_non_ascii(tweet.text))
            text.append(temp)

        query = '.'.join(text)
        data, emotion_sents, score, line_sentiment, text = processing_results(query)

        return render_template('projects/twitter.html', data=[data, emotion_sents, score, zip(text, line_sentiment)])
    else:
        return render_template('projects/twitter.html')

