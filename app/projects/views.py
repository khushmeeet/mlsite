from flask import render_template, Blueprint

projects_mod = Blueprint('projects', __name__, template_folder='templates', static_folder='templates')


@projects_mod.route('/line', methods=['GET', 'POST'])
def line():
    return render_template('projects/line.html')
