from flask import Flask, jsonify
from flask import request
from flask import abort
from flask_restful import Api, Resource
import forVK
import classifier
import classifier.vect_svc
import vk.exceptions as VKExceptions

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
        user = None

        try:
            user_res = forVK.get_user(user_id)
            if type(user_res) is VKExceptions.VkAPIError:
                raise user_res
            user = user_res[0]

            post_res = forVK.get_posts(user['id'])
            if type(post_res) is VKExceptions.VkAPIError:
                raise post_res

            user_posts = post_res['items']
            marked_posts_texts = set_post_toxicity(user_posts)
            toxicity_count = 0
            if len(marked_posts_texts) > 0:
                toxicity_count = sum(float(item['toxicity_mark']) for item in marked_posts_texts) / len(
                    marked_posts_texts)
            else:
                marked_posts_texts = None

            user['toxicity'] = toxicity_count
            return {'user': user, 'posts': marked_posts_texts, 'description': 'Пользователь успешно проанализирован'}
        except VKExceptions.VkAPIError as e:
            if e.code == 15:
                user['toxicity'] = 0
                user['is_closed'] = True
                return {'user': user, 'posts': None, 'description': 'Пользователь закрыт'}
            if e.code == 18:
                return {'user': None, 'posts': None, 'description': 'Пользователь забанен или удален'}
            if e.code == 113:
                return {'user': None, 'posts': None, 'description': 'Пользователь с таким идентификатором не найден'}
        except:
            abort(500, 'Something go wrong')


class Group(Resource):
    def get(self, group_id):
        group = None
        try:
            group_res = forVK.get_group(group_id)
            if type(group_res) is VKExceptions.VkAPIError:
                raise group_res
            group = group_res[0]
            
            post_res = forVK.get_posts(group_id)
            if type(post_res) is VKExceptions.VkAPIError:
                raise post_res
            group_posts = post_res['items']

            marked_posts_texts = set_post_toxicity(group_posts)
            toxicity_count = 0
            if len(marked_posts_texts) > 0:
                toxicity_count = sum(float(item['toxicity_mark']) for item in marked_posts_texts) / \
                                 len(marked_posts_texts)
            group['toxicity'] = toxicity_count
            return {'group': group, 'posts': marked_posts_texts, 'description': 'Группа успешно проанализирована'}
        except VKExceptions.VkAPIError as e:
            if e.code == 15:
                group['toxicity'] = 0
                group['is_closed'] = True
                return {'group': group, 'posts': None, 'description': 'Группа закрыта'}
            if e.code == 18:
                return {'group': None, 'posts': None, 'description': 'Группа забанена или удалена'}
            if e.code == 113:
                return {'group': None, 'posts': None, 'description': 'Группа с таким идентификатором не найдена'}
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
