import time
import vk_methods
import vk

# ----- id токсичных групп -----
# degradination
# kpopanti
# antidolboebi
# ------------------------------
access_token = ''

def get_group_comments(group_id):
    while True:
        post_id_list = vk_methods.get_posts_by_domain(group_id, 1, access_token)
        print(post_id_list)
        # a = str(post_id_list['items'][0]['id'])
        # comm = vk_methods.get_posts_comment(group_id, a, access_token, 1)
        # com_text = comm['items'][0]['text']
        # print(com_text)
        # time.sleep(5)

if __name__ == '__main__':
    get_group_comments("degradination")
