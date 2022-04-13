import json
import random
import math

def readjson():
    # Get details of the Dataset
    # Dataset link: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
    files = os.listdir(datapath)
    records = [["title", "records", "Keys_list"]]

    for file in files:
        m = datapath + file
        print(m)
        cnt = 0
        with open(m) as f:
            line = f.readline()
            data = json.loads(line)
            cnt = 1
            while line:
                line = f.readline()
                cnt += 1
            records.append([file, cnt, data.keys()])
            print(file + str(cnt))

    with open("records.csv", "w") as rec:
        for r in records:
            rec.write("%s\n" % r)

    return

def remove_zero_ratings(input_path, output_path):
    output_file = open(output_path, "w+")
    removed_count = 0
    with open(input_path, "r") as training_data:
        for current_line in training_data:
            data = json.loads(current_line)
            for current_review in data:
                if int(current_review["rating"]) != 0:
                    output_file.write(json.dumps(current_review))
                    output_file.write("\n")
                else:
                    removed_count += 1

    print(removed_count)
    output_file.close()

def create_test_train_split(input_path, output_train_path, output_test_path):
    with open(input_path, "r") as training_data:
        rating_indices = [[],[],[],[],[],[]]

        for current_line_index, review in enumerate(training_data):
            data = json.loads(review)
            del data["user_id"]
            del data["book_id"]
            del data["date_added"]
            rating_indices[int(data["rating"])].append(data)

        testing_data_indices = []
        training_data_indices = []
        for current_rank in range(1,6):
            print(len(rating_indices[current_rank]))
            random.shuffle(rating_indices[current_rank])
            testing_data_indices.extend(rating_indices[current_rank][:int(0.1 * len(rating_indices[current_rank]))])
            training_data_indices.extend(rating_indices[current_rank][int(0.1 * len(rating_indices[current_rank])):])

        print("==========")
        print(len(testing_data_indices))
        print(len(training_data_indices))

        random.shuffle(testing_data_indices)
        random.shuffle(training_data_indices)

        print("==========")
        x = 0
        with open(output_train_path,"w+") as sent_train_file:
            for current in training_data_indices:
                x += 1
                sent_train_file.write(json.dumps(current))
                sent_train_file.write("\n")

        print(x)
        with open(output_test_path,"w+") as sent_test_file:
            for current in testing_data_indices:
                x += 1
                sent_test_file.write(json.dumps(current))
                sent_test_file.write("\n")
        print(x)




# create_test_train_split("train_data-002.json")
# remove_zero_ratings("train_data-002.json", "train_no_zero.json")
create_test_train_split("train_no_zero.json", "train_sentiment.csv", "test_sentiment.csv")