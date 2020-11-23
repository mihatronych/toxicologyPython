import vk_methods
import pandas as pd
import re

# ----- id токсичных групп -----
# degradination
# kpopanti
# antidolboebi
# helisexual
#
# ------------------------------
access_token = ''


def get_group_comments(domain):
    post_id_list = vk_methods.get_posts_by_domain(domain, 100, access_token)
    group_id = post_id_list['items'][0]['owner_id']
    com_texts = []
    for i in range(len(post_id_list['items'])):
        post_id = str(post_id_list['items'][i]['id'])
        comments = vk_methods.get_posts(group_id, post_id, access_token, 100)
        for j in range(len(comments['items'])):
            if 'text' in comments['items'][j]:
                com_texts.append(re.sub('^\s+|\n|\r|\s+$', '', comments['items'][j]['text']))

    write_csv(com_texts)


def write_csv(comments):
    columns = ['comment']
    df = pd.DataFrame(comments, columns=columns)
    df.to_csv(r'vk_comments.csv', mode='a', header=True, index=False)


if __name__ == '__main__':
    group_domain = ''
    # get_group_comments(group_domain)
