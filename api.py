from flask import Flask
from flask import request
from flask import abort
import forVK
app = Flask(__name__)


@app.route('/toxicity_py/api/users/<string:id>', methods=['GET'])
def get_users(id):
    try:
        users = forVK.get_user(id)
        return {'users': users}
    except:
        abort(400)



@app.route('/toxicity_py/api/posts/<string:id>', methods=['GET'])
def get_posts(id):
    try:
        posts = forVK.get_posts(id)
        return {'posts': posts}
    except:
        abort(400)



@app.route('/toxicity_py/api/comments/<string:user_id>/<string:post_id>', methods=['GET'])
def get_comments(user_id, post_id):
    try:
        comments = forVK.get_posts_comment(user_id, post_id)
        return {'comments': comments}
    except:
        abort(400)

#post 411

@app.route('/toxicity_py/api/answers/<string:user_id>/<string:post_id>/<string:comments_id>', methods=['GET'])
def get_answers(user_id, post_id, comments_id):
    try:
        answers = forVK.get_comment_comments(user_id, post_id, comments_id)
        return {'answers': answers}
    except:
        abort(400)



@app.route('/toxicity_py/api/followers/<string:id>', methods=['GET'])
def get_followers(id):
    try:
        followers = forVK.get_users_followers(id)
        return {'followers': followers}
    except:
        abort(400)



@app.route('/toxicity_py/api/subscriptions/<string:id>', methods=['GET'])
def get_subscriptions(id):
    try:
        subscriptions = forVK.get_users_subscriptions(id)
        return {'subscriptions': subscriptions}
    except:
        abort(400)



@app.route('/toxicity_py/api/groups/<string:id>', methods=['GET'])
def get_groups(id):
    try:
        groups = forVK.get_group(id)
        return {'groups': groups}
    except:
        abort(400)


@app.route('/toxicity_py/api/members/<string:id>', methods=['GET'])
def get_members(id):
    try:
        members = forVK.get_groups_members(id)
        return {'members': members}
    except:
        abort(400)


@app.route('/toxicity_py/api/message', methods=['POST'])
def get_message():
    return "Hello!"


@app.route('/toxicity_py/api/messages', methods=['POST'])
def get_messages():
    return "Hello!"


if __name__ == '__main__':
    app.run(debug=True)
