from flask import Flask, jsonify
from flask import request
from flask import abort
import forVK
import classifier
import classifier.xgb
import vect_svc

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


# post 411

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


@app.route('/toxicity_py/api/message', methods=['POST', 'GET'])
def get_message():
    if request.method == 'POST':
        data = request.json['messages']
        labeled_messages = classifier.xgb.classifier([data])
        result = []
        for comment, toxic in labeled_messages:
            result.append({
                'message': comment,
                'toxic': str(toxic[1])
            })
        print(result)
        return jsonify(result)
    else:
        abort(400)


@app.route('/toxicity_py/api/messages', methods=['POST', 'GET'])
def get_messages():
    if request.method == 'POST':
        data = request.json['messages']
        labeled_messages = classifier.xgb.classifier(data)
        result = []
        for comment, toxic in labeled_messages:
            result.append({
                'message': comment,
                'toxic': str(toxic[1])
            })
        print(result)
        return jsonify(result)
    else:
        abort(400)


@app.route('/toxicity_py/api/rude_feature_extraction', methods=['POST', 'GET'])
def get_rude_feature_extraction():
    if request.method == 'POST':
        comment = request.json['comment']
        prop_neg_to_text = vect_svc.rude_feature_extraction(comment)
        return jsonify(prop_neg_to_text)
    else:
        abort(400)


@app.route('/toxicity_py/api/some_spicy_features_extraction', methods=['POST', 'GET'])
def get_some_spicy_features_extraction():
    if request.method == 'POST':
        comment = request.json['comment']
        per_c, loc_c, org_c, pos_c, neg_c, neu_c, sp_c, sk_c = vect_svc.some_spicy_features_extraction(comment)
        #neg_c, neu_c, sp_c, sk_c
        return {
            'per_c': per_c,
            'loc_c': loc_c,
            'org_c': org_c,
            'pos_c': pos_c,
            'neg_c': neg_c,
            'neu_c': neg_c,
            'sp_c': sp_c,
            'sk_c': sk_c
        }
    else:
        abort(400)


if __name__ == '__main__':
    app.run(debug=True)
