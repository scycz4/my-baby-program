import csv
import os
from time import sleep

import numpy
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.utils import Bunch
from sklearn.neural_network import MLPClassifier

from audio_split import custom_audio_split
from test.test1 import model_create, own_data
from collect_features import read_wav_files, extract_wav_feature


def read_all_file1(record_files, head=0):
    if os.path.isdir(record_files):
        files = os.listdir(record_files)
        for file in files:
            record_path = record_files + "/" + file
            if os.path.isdir(record_path):
                read_all_file1(record_path, head)
            else:
                # print(record_path)
                list = file.split('.')
                if list[len(list) - 1] == "pk" or list[len(list) - 1] == "npy":
                    continue
                else:
                    test = own_data(record_path, head)
                    head = 1

    else:
        # print(record_files)
        test = own_data(record_files, head)
        head = 1


global list_store
list_store = []


def read_all_file(record_files, clf):
    if os.path.isdir(record_files):
        files = os.listdir(record_files)
        l1 = None
        positive_num = 0
        total = 0
        for file in files:
            if l1 is None:
                l1 = file.split('_')[0] + "_" + file.split('_')[1]
                total = 1
                positive_num = 0
            elif l1 != file.split('_')[0] + "_" + file.split('_')[1]:
                list_store.append([l1 + "." + list[len(list) - 1], positive_num / total])
                l1 = file.split('_')[0] + "_" + file.split('_')[1]
                total = 1
                positive_num = 0
            else:
                total += 1
            record_path = record_files + "/" + file
            print(record_path)
            if os.path.isdir(record_path):
                read_all_file(record_path)
            else:
                # print(record_path)
                list = file.split('.')
                if list[len(list) - 1] == "pk" or list[len(list) - 1] == "npy":
                    continue
                else:
                    fs_list, signals, rmse_list = extract_wav_feature(record_path)
                    rmse_list = numpy.reshape(rmse_list, [-1, 1])
                    result = clf.predict(rmse_list)
                    if result[0] == 1:
                        positive_num += 1
                    # print(file + ":" + str(result))
    else:
        # print(record_files)
        fs_list, signals, rmse_list = extract_wav_feature(record_files)
        rmse_list = numpy.reshape(rmse_list, [-1, 1])
        result = clf.predict(rmse_list)
        # print(record_files + ":" + str(result))


def attempt():
    # X_train,x_test,Y_train,y_test = model_create("property.cfg", "split_audio")


    # fs_list, signals, rmse_list = read_wav_files("duration_limit/")
    # rmse_list = numpy.reshape(rmse_list, [-1, 1])
    #
    # clf = OneClassSVM(gamma='auto').fit(rmse_list)
    #
    # # custom_audio_split("./audio_need_predict/origin_audio/","./audio_need_predict/convert2wav/",save_chunks_file_folder=("./audio_need_predict/detected_split1/","./audio_need_predict/detected_split2/"),audio_pure_wav="./audio_need_predict/dropNoice/",output_limitation="./audio_need_predict/duration_limit/")
    #
    # record_files = "./duration_limit"
    # read_all_file(record_files, clf)
    # # print(list_store)
    #
    # for i in list_store:
    #     if i[1] < 0.4:
    #         i[1] = 0
    #     else:
    #         i[1] = 1
    #
    # read_all_file1("duration_limit/")



    dataset = load_my_dataset()

    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(100,), learning_rate='adaptive',random_state=1)
    sleep(10)
    clf.fit(X_train, y_train)
    # print(clf.score(X_test,y_test))


    predictions_train=clf.predict(X_train)
    predictions_test=clf.predict(X_test)
    predictions_total=clf.predict(X)
    train_score=accuracy_score(predictions_train,y_train)
    print("score on train data: ",train_score)
    test_score=accuracy_score(predictions_test,y_test)
    print("score on test data: ",test_score)
    total_score=accuracy_score(predictions_total,y)
    print("score on total data: ",total_score)
    print(classification_report(predictions_total,y))


    # fs_list, signals, rmse_list = read_wav_files("duration_limit/")
    # print(rmse_list)
    # X=rmse_list


def load_my_dataset():
    with open(r'temp.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            features.pop(0)
            feature=[float(num) for num in features]
            data.append(feature)
            target.append(int(label))

        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)


if __name__ == "__main__":
    attempt()
