import os

from sklearn.svm import OneClassSVM
from audio_split import custom_audio_split
from test.test1 import model_create, own_data


def read_all_file(record_files, clf):
    if os.path.isdir(record_files):
        files = os.listdir(record_files)
        for file in files:
            record_path = record_files + "/" + file
            if os.path.isdir(record_path):
                read_all_file(record_path)
            else:
                # print(record_path)
                list = file.split('.')
                if list[len(list) - 1] == "pk":
                    continue
                else:
                    test = own_data(record_path)

                    result = clf.predict(test)
                    print(record_path + ":" + str(result[0]))
    else:
        # print(record_files)
        test = own_data(record_files)

        result = clf.predict(test)
        print(record_files + ":" + str(result[0]))


def attempt():
    X_train, x_test, Y_train, y_test = model_create("property.cfg", "split_audio")
    clf = OneClassSVM(gamma='auto').fit(X_train)
    custom_audio_split("./audio_need_predict/origin_audio","./audio_need_predict/convert2wav",save_chunks_file_folder=("./audio_need_predict/detected_split1","./audio_need_predict/detected_split2"),audio_pure_wav="./audio_need_predict/dropNoice",output_limitation="./audio_need_predict/duration_limit")
    record_files = "./audio_need_predict/duration_limit"
    read_all_file(record_files, clf)


if __name__ == "__main__":
    attempt()
