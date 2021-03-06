from flask import Flask, url_for
from .home.views import home_mod
from .liner.views import liner_mod
from .doc.views import doc_mod
from .twitter.views import twitter_mod
from .team.views import team_mod
import os


UPLOAD_FOLDER = './uploads/'
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.register_blueprint(home_mod)
app.register_blueprint(liner_mod)
app.register_blueprint(doc_mod)
app.register_blueprint(twitter_mod)
app.register_blueprint(team_mod)


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
