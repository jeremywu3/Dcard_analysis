# coding=utf-8
import requests
import json
import os

def crawler(board):
    if not os.path.isdir(board):
        os.mkdir(board)
    L = 1
    total = 0
    accumulate = 0
    acc_posts = []
    path = "http://www.dcard.tw"
    path += "/_api/forums/{0}/posts".format(board)
    pre_id = dict()
    while L > 0:
        res = requests.get(path,params=pre_id)
        posts = json.loads(res.text)
        L = len(posts)
        total += L
        accumulate += L
        acc_posts += posts
        if accumulate == 150:
            filename = "{0}/".format(board) + str(acc_posts[0]['id']) + ".json"
            with open(filename,"w") as fp:
                json.dump(acc_posts,fp,ensure_ascii=False)
            accumulate = 0
            acc_posts = []
            print("Store %s" % (filename))
        pre_id['before'] = posts[-1]['id']
        print(total)

    print(total)

if __name__ == "__main__":
    crawler("mood")

