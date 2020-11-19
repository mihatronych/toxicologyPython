import vk


def get_user(user_id, access_token):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.users.get(user_id=user_id, fields="photo_200, can_see_all_posts")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_posts(id, count, access_token):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.wall.get(owner_id=id,  count=str(count), filter="all")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_posts_by_domain(domain, count, access_token):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.wall.get(domain=domain,  count=str(count), filter="all")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_posts_comment(id, post_id, access_token, count=10):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.wall.getComments(owner_id=id, post_id=post_id, count=str(count), sort="asc", preview_length=0)
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_comment_comments(id, post_id, comment_id, access_token, count=10):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.wall.getComments(owner_id=id, comment_id=comment_id, post_id=post_id, count=str(count), sort="asc", preview_length=0)
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_users_followers(user_id, access_token, count=100):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.users.getFollowers(user_id=user_id, count=count, fields="photo_200, is_friend, wall_comments")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_users_subscriptions(user_id, access_token, count=100):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.users.getSubscriptions(user_id=user_id, count=count)
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_group(group_id, access_token):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.groups.getById(group_id=group_id, fields="description, can_see_all_posts")
    except vk.exceptions.VkAPIError as e:
        return e.message


def get_groups_members(group_id, access_token, count=1000):
    try:
        session = vk.Session(access_token=access_token)
        vk_api = vk.API(session, v="5.126")
        return vk_api.groups.getMembers(group_id=group_id, sort="id_asc", count=count)
    except vk.exceptions.VkAPIError as e:
        return e.message

if __name__ == '__main__':
    access_token = ''
    # print(get_user(1, access_token))
    # print(get_posts(1, 20, access_token))
    # print(get_posts_by_domain("degradination", 20, access_token))
    # print(get_posts_comment(1, 2442097, access_token, 20))
    # print(get_comment_comments(1, 2442097, 2442108, access_token, 20))
    # print(get_users_followers(1, access_token, 100))
    # print(get_users_subscriptions(1, access_token, 100))
    # print(get_group(1, access_token))
    # print(get_groups_members(1, access_token, 100))

