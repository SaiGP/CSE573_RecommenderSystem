import json
import os.path
import numpy as np
import pandas as pd


def user_rating():
    print("Create mat ...")
    users = np.loadtxt("./user.out")
    vh = np.loadtxt("./vh.out")
    mat = np.matmul(users, vh)
    mat1 = 5 * np.linalg.norm(mat)
    np.savetxt("user_item_sim.out", mat)
    np.savetxt("ui_sim.out", mat1)

    print(mat.shape)
    return

def data_abs():
    dfp = open("data.out", "r")
    data = json.load(dfp)
    items = data["items"]
    user_item = {}
    for i, user in enumerate(data["users"]):
        if user in user_item.keys():
            user_item[user].append(data["items"][i])
        else:
            user_item[user] = [data["items"][i]]

    fp = open("user_item.out", "w")
    json.dump(user_item, fp)
    return


def itemTitle():
    dfp = open("data.out", "r")
    data = json.load(dfp)
    items = data["items"]
    item_title = {}
    bfp = open("../books_list.json", "r")
    books = json.load(bfp)

    for book in books:
        if book["book_id"] in items:
            item_title[book["book_id"]] = book["title"] # "title"

    fp = open("item_title.json", "w")
    json.dump(item_title, fp)

    fp.close()
    bfp.close()
    dfp.close()
    return


def get_user_list():
    df = pd.read_pickle("pivot.pk")
    print(df.index)
    # np.savetxt("col_name.out", df.columns, fmt='%s')
    input("press to go...")
    np.savetxt("row_name.out", df.index, fmt='%s')
    return list(df.index), list(df.columns)


def get_recommendations(userid, mat):
    user_list = list(np.loadtxt("row_name.out", dtype='str'))
    item_ids = np.loadtxt("col_name.out", dtype='str')
    # print(user_list)
    # print(userid)

    ind = user_list.index(userid)
    if ind > mat.shape[0]:
        return
    lst = mat[ind]
    ind_10 = lst.argsort()[:10]
    for i in ind_10:
        if i > item_ids.shape[0]:
            ind_10 = ind_10[ind_10 != i]
    items = item_ids[ind_10]
    rating = np.round(lst[ind_10], decimals=2)
    #print(items)
    # print(rating)

    fp = open("item_title.json", "r")
    item_title = json.load(fp)
    book_list = [item_title[str(item)] for item in items]

    recos = [list(d) for d in zip(book_list, rating)]

    print(book_list)

    return


def prediction():
    if not os.path.exists("ui_sim.out"):
        user_rating()

    print("Loading mat ...")
    sim = open("ui_sim.out", "r")
    mat = np.loadtxt(sim)
    # mat = ((mat1 - np.min(mat1))/np.ptp(mat1))*5
    # np.savetxt("ui_sim.out", mat)
    print(mat.shape)

    if not os.path.exists("user_item.out"):
        data_abs()

    ufp = open("user_item.out", "r")
    user_items = json.load(ufp)
    ufp.close()

    if not os.path.exists("item_title.json"):
        itemTitle()

    ifp = open("item_title.json", "r")
    item_title = json.load(ifp)
    ifp.close()

    user_pred_list = ['390250e5e7a117885d27cf51b353ec42', 'a55483e34c0c2fbd4cc36b17edf0487e', 'fdc7044fa37a0d1c268e01ef4646e17e', '7057cfddf7bbe8ca19a30177f20bd78e',
                      '97e2ce2141fa1c880967d78aec3c14fa', '0d9ba699a18dd5858690bb55aa3cd6d5', '1b42e81ddcb644460d574f64f45b9c44', '93cc3a086b687cec69d860ebfa3746c8',
                      'b78a9143ca2f0c4c7361694dc6cb0500', '62736c3089a1d8dc7035d790e4f6e229', 'eb59adf1a28f90b365bd1dc54c0c9487', '057e6965cbbe05838ef86d1ee8eef1ea',
                      'edac8a00c488d38c853a9842e0af51c3']

    for u in user_pred_list:
        print(u)
        get_recommendations(u, mat)

    while True:
        userid = input("Enter a user ID!!>>")
        if userid == "N" or userid == "n":
            break
        get_recommendations(userid, mat)


# get_user_list()
prediction()
