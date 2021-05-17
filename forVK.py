import vk
import os
from dotenv import load_dotenv
import vk.exceptions

load_dotenv()

session = vk.Session(access_token=os.getenv('ACCESS_TOKEN'))
vk_api = vk.API(session, v=os.getenv('VK_API_VERSION'))


# получить пользователя по ID
def get_user(user_id):
    try:
        return vk_api.users.get(user_ids=user_id, fields="photo_200, can_see_all_posts")
    except vk.exceptions.VkAPIError as e:
        return e.message


# получить группу по ID
def get_group(group_id):
    try:
        return vk_api.groups.getById(group_ids=group_id, fields="description, can_see_all_posts")
    except vk.exceptions.VkAPIError as e:
        return e.message


# получить пост по его ID
def get_post(post_id):
    try:
        return vk_api.wall.getById(posts=post_id)
    except vk.exceptions.VkAPIError as e:
        return e.message


# получить получить посты владельца (группы или пользователя) по его ID
def get_posts(owner, count=100):
    try:
        if str.isdigit(str(owner)):
            return vk_api.wall.get(owner_id=owner, count=str(count), filter="all")
        else:
            return vk_api.wall.get(domain=owner, count=str(count), filter="all")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_posts_comments(owner, post_id, count=10):
    try:
        if str.isdigit(str(owner)) is False:
            owner = vk_api.wall.get(domain=owner, count=str(count), filter="all")['items'][0]['owner_id']
        return vk_api.wall.getComments(owner_id=owner, post_id=post_id, count=str(count), sort="asc", preview_length=0)
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_comment_comments(owner, post_id, comment_id, count=10):
    try:
        if str.isdigit(str(owner)) is False:
            owner = vk_api.wall.get(domain=owner, count=str(count), filter="all")['items'][0]['owner_id']
        return vk_api.wall.getComments(owner_id=owner, comment_id=comment_id, post_id=post_id, count=str(count),
                                       sort="asc", preview_length=0)
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_users_followers(user_id, count=100):
    try:
        if str.isdigit(str(user_id)) is False:
            user_id = vk_api.users.get(user_ids=user_id)[0]['id']
        return vk_api.users.getFollowers(user_id=user_id, count=count, fields="photo_200, is_friend, wall_comments")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_users_subscriptions(user_id, count=100):
    try:
        if str.isdigit(str(user_id)) is False:
            user_id = vk_api.users.get(user_ids=user_id)[0]['id']
        return vk_api.users.getSubscriptions(user_id=user_id, count=count)
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_groups_members(group_id, count=1000):
    try:
        if str.isdigit(str(group_id)) is False:
            group_id = vk_api.groups.getById(group_ids=group_id)[0]['id']
        return vk_api.groups.getMembers(group_id=group_id, sort="id_asc", count=count)
    except vk.exceptions.VkAPIError as e:
        return e.message


if __name__ == '__main__':
    access_token = ''
    # (get_user('1', access_token))
    # print(get_posts("ismaakova", 20, access_token))
    # print(get_posts_comment("ismaakova", 411, access_token, 20))
    # print(get_comment_comments(1, 2442097, 2442108, access_token, 20))
    # print(get_users_followers("ismaakova", access_token, 100))
    # print(get_users_subscriptions(1, access_token, 100))
    # print(get_group(1, access_token))
    # print(get_groups_members('thecode.media', access_token, 100))
