import argparse
from pathlib import Path

import numpy as np
from utils import Util
from rbm import RBM
import json
import csv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--num_hid', type=int, default=64,
                    help='Number of hidden layer units (latent factors)')
parser.add_argument('--user', type=str, default=22,
                    help='user id to recommend books \
                    to (not all ids might be present)')
parser.add_argument('--data_dir', type=str, default='data', required=True,
                    help='path to dataset')
parser.add_argument('--rows', type=int, default=200000,
                    help='number of rows to be used for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--free_energy', type=bool, default=False,
                    help='Export free energy plot')
parser.add_argument('--verbose', type=bool, default=False,
                    help='Display info after each epoch')
args = parser.parse_args()

def main():



    # print('Converting review_data.json to csv ')
    # base_path = Path(__file__).parent
    # print('Base path : ', base_path)
    # records = [["user_id", "records", "rating", "review_text", "date_added"]]
    #
    # with open((base_path / "data/review_data.json").resolve()) as json_file:
    #     line = json_file.readline()
    #     data = json.loads(line)
    #     cnt = 1
    #     while line:
    #         line = json_file.readline()
    #         cnt += 1
    #         records.append(data)
    #     with open("review_data.csv", "w") as rec:
    #         for r in records:
    #             rec.write("%s\n" % r)




     # for file in files:
    #     m = datapath + file
    #     print(m)
    #     cnt = 0
    #     with open(m) as f:
    #         line = f.readline()
    #         data = json.loads(line)
    #         cnt = 1
    #         while line:
    #             line = f.readline()
    #             cnt += 1
    #         records.append([file, cnt, data.keys()])
    #         print(file + str(cnt))
    # with open("records.csv", "w") as rec:
    #     for r in records:
    #         rec.write("%s\n" % r)

    # print('Removing reviews and date_added from review_list.csv')
    # base_path = Path(__file__).parent
    # ratings = pd.read_csv((base_path / "data/reviews_list.csv").resolve())
    # keep_col = ['user_id', 'book_id', 'rating']
    # new_f = ratings[keep_col]
    # new_f.to_csv((base_path / "data/new_ratings.csv").resolve(), index=False)

    # print('Creating to_read_books.csv file  :')
    # util = Util()
    # # ratings, to_read, books, book_ids = util.read_data(dir)
    # ratings, book_ids = util.read_data(dir)
    # book_ids_all = book_ids['book_id'].tolist()
    # ratings.sort_values(["user_id"],
    #                     axis=0,
    #                     ascending=[True],
    #                     inplace=True)
    # # print(ratings.groupby('user_id')['book_id'].apply(list))
    #
    # header = ['user_id', 'book_id']
    # base_path = Path(__file__).parent
    # with open((base_path / "data/to_read_books.csv").resolve(), 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     # write the header
    #     writer.writerow(header)
    #     # write the data
    #     for record in ratings.groupby('user_id'):
    #         # print('User : ', record[0], 'Books : ', record[1]['book_id'].tolist())
    #         for book in record[1]['book_id'].tolist():
    #             data=[]
    #             data.append(record[0])
    #             data.append(book)
    #             writer.writerow(data)
    #             # print('User : ', record[0], 'Book : ', book)
    #     # writer.writerow(data)







    util = Util()
    dir = args.data_dir
    rows = args.rows
    ratings, to_read, books, book_ids = util.read_data(dir)
    book_ids_all = book_ids['book_id'].tolist()
    # print('Book IDs : ', book_ids_all)
    ratings = util.clean_subset(ratings, rows)
    num_vis = len(ratings)
    free_energy = args.free_energy
    train = util.preprocess(ratings)
    valid = None
    if free_energy:
        train, valid = util.split_data(train)
    H = args.num_hid
    user = args.user
    alpha = args.alpha
    w = np.random.normal(loc=0, scale=0.01, size=[num_vis, H])
    rbm = RBM(alpha, H, num_vis)
    epochs = args.epochs
    batch_size = args.batch_size

    v = args.verbose
    reco, prv_w, prv_vb, prv_hb = rbm.training(train, valid, user,
                                                epochs, batch_size,
                                                free_energy, v)
    unread, read = rbm.calculate_scores(ratings, books,
                                        to_read, reco, user)
    rbm.export(unread, read)

if __name__ == "__main__":
    main()
