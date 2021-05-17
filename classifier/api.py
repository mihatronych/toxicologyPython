from flask import Flask, jsonify
from flask import request
from flask import abort
from flask_restful import Api, Resource
import forVK
import classifier
import classifier.vect_svc

app = Flask(__name__)
api = Api(app)


def analyse_texts(texts_list):
    labeled_messages = classifier.vect_svc.classifier(texts_list)
    result = []
    for comment, toxic in labeled_messages:
        result.append(str(toxic[0]))
    return result


def set_post_toxicity(posts):
    clear_posts = [post for post in posts if post['text'] != '']
    if len(clear_posts) > 0:
        marked_posts_texts = analyse_texts([post['text'] for post in clear_posts])
        for index, post in enumerate(clear_posts):
            clear_posts[index]['toxicity_mark'] = marked_posts_texts[index]
    return clear_posts


class User(Resource):
    def get(self, user_id):
        try:
            user = forVK.get_user(user_id)[0]
            user_posts = forVK.get_posts(user['id'])['items']
            marked_posts_texts = set_post_toxicity(user_posts)
            toxicity_count = 0
            if len(marked_posts_texts) > 0:
                toxicity_count = sum(float(item['toxicity_mark']) for item in marked_posts_texts) / len(
                    marked_posts_texts)
            else:
                marked_posts_texts = None
            user['toxicity'] = toxicity_count
            # user['is_closed'] = 1 if user['is_closed'] else 0

            return {'user': user, 'posts': marked_posts_texts}
        except:
            abort(500, 'Something go wrong')


class Group(Resource):
    def get(self, group_id):
        try:
            group = forVK.get_group(group_id)[0]
            group_posts = forVK.get_posts(group_id)['items']
            marked_posts_texts = set_post_toxicity(group_posts)
            toxicity_count = 0
            if len(marked_posts_texts) > 0:
                toxicity_count = sum(float(item['toxicity_mark']) for item in marked_posts_texts) / \
                                 len(marked_posts_texts)
            group['toxicity'] = toxicity_count
            return {'group': group, 'posts': marked_posts_texts}
        except:
            abort(500, 'Something go wrong')


class Post(Resource):
    def get(self, post_id):
        try:
            post = forVK.get_post('-'+post_id.replace('-', '_'))[0]
            print(post)
            marked_post = set_post_toxicity([post])
            owner = forVK.get_group(str(post['owner_id']).replace('-', ''))[0]
            return {'post': marked_post, 'owner': owner}
        except:
            abort(500, 'Something go wrong')


api.add_resource(User, "/api/user/<string:user_id>")
api.add_resource(Group, "/api/group/<string:group_id>")
api.add_resource(Post, "/api/post/<string:post_id>")

if __name__ == '__main__':
    app.run(debug=True)
