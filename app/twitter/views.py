from flask import Blueprint, request, render_template
from ..load import processing_results
import tweepy
import string
from memory_profiler import profile


twitter_mod = Blueprint('twitter', __name__, template_folder='templates', static_folder='static')

ascii_chars = set(string.printable)
ascii_chars.remove(' ')


def takeout_non_ascii(s):
    return list(filter(lambda x: x not in ascii_chars, s))

@profile
@twitter_mod.route('/twitter', methods=['GET', 'POST'])
def twitter():
    if request.method == 'POST':
        auth = tweepy.OAuthHandler('hXJ8TwQzVya3yYwQN1GNvGNNp', 'diX9CFVOOfWNli2KTAYY13vZVJgw1sYlEeOTxsLsEb2x73oI8S')
        auth.set_access_token('2155329456-53H1M9QKqlQbEkLExgVgkeallweZ9N74Aigm9Kh',
                              'waDPwamuPkYHFLdVNZ5YF2SNWuYfGHDVFue6bEbEGjTZb')

        api = tweepy.API(auth)

        public_tweets = api.search(request.form['topic'], lang='hi', rpp=20)
        text = []
        for tweet in public_tweets:
            temp = ''.join(takeout_non_ascii(tweet.text))
            text.append(temp)

        query = '.'.join(text)
        data, emotion_sents, score, line_sentiment, text = processing_results(query)
        # del data, text

        return render_template('projects/twitter.html', data=[data, emotion_sents, score, zip(text, line_sentiment)])
    else:
        return render_template('projects/twitter.html')

