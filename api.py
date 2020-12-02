#как выглядит запрос http://[hostname]/toxicity/api
from flask import Flask
from flask import request
from flask import abort
import forVK
app = Flask(__name__)


@app.route('/toxicity_py/api/users/<int:id>', methods=['GET'])
def get_users(id):
    if not request.json or not 'title' in request.json:
        abort(400)
    users = forVK.get_user(id)
    return "Hello!"


@app.route('/toxicity_py/api/posts/<int:id>', methods=['GET'])
def get_posts(id):
    return "Hello!"


@app.route('/toxicity_py/api/comments/<int:user_id>/<int:post_id>', methods=['GET'])
def get_comments(user_id, post_id):
    return "Hello!"


@app.route('/toxicity_py/api/answers/<int:user_id>/<int:comments_id>', methods=['GET'])
def get_answers(user_id, comments_id):
    return "Hello!"


@app.route('/toxicity_py/api/followers/<int:id>', methods=['GET'])
def get_followers(id):
    return "Hello!"


@app.route('/toxicity_py/api/subscriptions/<int:id>', methods=['GET'])
def get_subscriptions(id):
    return "Hello!"


@app.route('/toxicity_py/api/groups/<int:id>', methods=['GET'])
def get_groups(id):
    return "Hello!"


@app.route('/toxicity_py/api/members/<int:id>', methods=['GET'])
def get_members(id):
    return "Hello!"


@app.route('/toxicity_py/api/message', methods=['POST'])
def get_message():
    return "Hello!"


@app.route('/toxicity_py/api/messages', methods=['POST'])
def get_messages():
    return "Hello!"


if __name__ == '__main__':
    app.run(debug=True)
