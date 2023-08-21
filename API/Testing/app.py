from distutils.log import debug
from fileinput import filename
from flask import *

app = Flask(__name__)


@app.route('/')
def main():
    return "index.html"


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        # f.save(f.filename)
        data = request.form
        return {'data' : f.filename}, 201


if __name__ == '__main__':
    app.run(debug=True)
