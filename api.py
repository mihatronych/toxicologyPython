#как выглядит запрос http://[hostname]/toxicity/api
from flask import Flask
app = Flask(__name__)

@app.route('/toxicity/api/comments')
def index():
    return "Hello!"

if __name__ == '__main__':
    app.run(debug=True)
