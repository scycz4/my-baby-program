import os
from configparser import ConfigParser
from time import sleep

import librosa
import numpy as np
import timbral_models
from timbral_models import timbral_sharpness


def feature_extract(records, timbral_models=None):
    y, sr = librosa.load(records)
    rmse = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    mfcc = librosa.feature.mfcc(y=y)
    spec_flat = librosa.feature.spectral_flatness(y=y)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    y, sr = librosa.load(records, res_type='kaiser_fast')
    stft = librosa.stft(y)
    power_spectrum = np.square(np.abs(stft))
    bins = librosa.fft_frequencies(sr=sr)
    # loudness = librosa.perceptual_weighting(power_spectrum, bins)
    # loudness = librosa.db_to_amplitude(loudness)
    # loudness = np.log(np.mean(loudness, axis=0) + 1e-5)

    # sharpness =timbral_sharpness(records)

    string = "file name:" + records + "     均方根能量:" + str(rmse) + "          过零率:" + str(zcr) + "           谱质心:" + str(
        spec_cent) + "           mfcc:" + str(mfcc) + "           频谱平坦度:" + str(spec_flat) + "        基音频率:" + str(
        f0) + "," + str(voiced_flag) + "," + str(voiced_probs)+" "
    text_save(string)


def read_all_file(record_files):
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

                    feature_extract(record_path)
    else:
        # print(record_files)
        feature_extract(record_files)


def text_save(data):
    file = open("baby data.txt", 'a')
    data = data + "\n"
    file.write(data)
    file.close()


if __name__ == "__main__":
    # read property file
    config = ConfigParser()
    config.read('property.cfg')
    # get the path of mp3 file
    # mp3 = config.get("mp3files", "mp3file1")
    mp3files = config.options("mp3files")
    sleep(1)
    print("file name       " + "均方根能量        " + "过零率         " + "谱质心          ", end="")
    print("mfcc         " + "频谱平坦度          " + "频谱通量         " + "基音频率       ", end="")
    print("响度          " + "尖锐度")
    for i in mp3files:
        record_files = config.get("mp3files", i)
        read_all_file(record_files)
