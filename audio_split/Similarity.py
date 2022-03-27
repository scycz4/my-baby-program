# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:51:38 2020

@author: yoona
"""
import os
import sys
import wave
import numpy as np
# import struct
from pydub import AudioSegment
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioFeatureExtraction as afe
import eyed3
import random
import math


def Features(path, mode):
    x = wave.open(path)
    params = x.getparams()
    print(params)

    if params[0] != 2:
        raise ValueError('通道数不等于2')

    strData = x.readframes(params[3])
    w = np.frombuffer(strData, dtype=np.int16)
    w = np.reshape(w, [params[3], params[0]])

    k = 4
    ii = 0
    w_decrease = [[], []]

    if mode == 'second':
        #   降低音频分辨率
        for kk in (0, 1):
            while ii < len(w[:, kk]):
                if ii + k < len(w[:, kk]):
                    w_decrease[kk].append(np.mean(w[ii:ii + k, kk]))
                else:
                    w_decrease[kk].append(np.mean(
                        w[ii:len(w[:, kk]) + 1, kk]))
                ii = ii + k
        w = w_decrease

    eigen_vector_0 = afe.mtFeatureExtraction(
        w[:, 0], params[2], 30.0, 30.0, 2, 2)
    eigen_vector_1 = afe.mtFeatureExtraction(
        w[:, 1], params[2], 30.0, 30.0, 2, 2)

    return eigen_vector_0, eigen_vector_1


def read_wave(wav_path):
    w = wave.open(wav_path)
    params = w.getparams()
    #    print(params)
    #   声道数、量化位数（byte)、采样频率、采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]

    #   文件时间
    t = np.arange(0, nframes) * (1 / framerate)
    strData = w.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels])  # 双通道数

    #    plot the wave
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t, waveData[:, 0])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Ch-1 wavedata")
    plt.grid('on')  # 标尺，on：有，off:无
    plt.subplot(4, 1, 3)
    plt.plot(t, waveData[:, 1])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Ch-2 wavedata")
    plt.grid('on')  # 标尺，on：有，off:无
    plt.show()


def similarity(v1, v2):
    #   计算平均相似度
    temp = []
    sim = []
    p = 0
    q = 1

    for ii in range(v1.shape[0]):
        for jj in range(v1.shape[1]):
            if v1[ii, jj] != 0 or v2[ii, jj] != 0:
                temp.append((1 -
                             abs(v1[ii, jj] - v2[ii, jj]) / max(abs(v1[ii, jj]), abs(v2[ii, jj]))))
                q += 1
        sim.append(np.mean(temp[p:q]))
        p = q
    print(sim)
    return sim


def compute_chunk_features(mp3_path):
    # =============================================================================
    # 计算相似度第一步
    # =============================================================================
    #   获取歌曲时长
    # mp3Info = eyed3.load(mp3_path)
    # time = int(mp3Info.info.time_secs)
    # print(time)
    # tail, track = os.path.split(mp3_path)
    #
    # #   创建两个文件夹
    # dirct_1 = tail + r'\wavSession'
    # dirct_2 = tail + r'\wavBlock'
    # if not os.path.exists(dirct_1):
    #     os.makedirs(dirct_1)
    # if not os.path.exists(dirct_2):
    #     os.makedirs(dirct_2)
    #
    # #   获取歌曲名字
    # song_name = track.split('.')
    #   转换格式
    # wav_all_path = os.path.join(tail, song_name[0] + '.wav')
    sound = AudioSegment.from_file(mp3_path, format='wav')
    # sound.export(wav_all_path, format='wav')
    # read_wave(wav_all_path)
    # #   划分音频
    # gap = 4
    # diff = time / 10 - 8
    # start_time = 0
    # end_time = math.floor(diff)
    vector_0 = np.zeros((10, 68))
    vector_1 = np.zeros((10, 68))
    info = []  # 记录片段开始时间点

    for jj in range(5):
        # wav_name = song_name[0] + str(jj) + '.wav'
        # wav_path = os.path.join(tail, 'wavSession', wav_name)
        # #       随机产生四秒片段
        # rand_start = random.randint(start_time, end_time)
        # blockData = sound[rand_start * 1000:(rand_start + gap) * 1000]
        # ##       音频切片,时间的单位是毫秒
        # #        blockData = sound[start_time*1000:end_time*1000]
        # blockData.export(wav_path, format='wav')
        # eigVector_0, eigVector_1 = Features(wav_path, [])

        eigVector_0, eigVector_1 = Features(sound, [])


        print(jj)  # 标记程序运行进程
        #       得到一个片段的特征向量
        vector_0[jj, :] = np.mean(eigVector_0[0], 1)
        vector_1[jj, :] = np.mean(eigVector_1[0], 1)
        #       迭代
        diff = diff + 4
        info.append((start_time, end_time))
        start_time = end_time
        end_time = math.floor(start_time + diff)

    #   承上启下
    end_time = start_time

    for kk in range(5, 10):
        #       迭代
        diff = diff - 4
        info.append((start_time, start_time + diff))
        start_time = end_time
        end_time = math.floor(start_time + diff)
        wav_name = song_name[0] + str(kk) + '.wav'
        wav_path = os.path.join(tail, 'wavSession', wav_name)

        #       随机产生四秒片段
        rand_start = random.randint(start_time, end_time)
        blockData = sound[rand_start * 1000:(rand_start + gap) * 1000]
        #        blockData = sound[start_time*1000:end_time*1000]
        blockData.export(wav_path, format='wav')
        eigVector_0, eigVector_1 = Features(wav_path, [])
        print(kk)  # 标记程序运行进程
        #       得到一个片段的特征向量

        vector_0[kk, :] = np.mean(eigVector_0[0], 1)
        vector_1[kk, :] = np.mean(eigVector_1[0], 1)

    return vector_0, vector_1, info  # 双通道各自的特征向量


def Compute_Bolck_Features(info, mp3_path):
    # =============================================================================
    # 计算相似度第二步
    # =============================================================================
    #   获取歌曲时长
    mp3Info = eyed3.load(mp3_path)
    time = int(mp3Info.info.time_secs)
    print(time)
    #   获取歌曲名字
    tail, track = os.path.split(mp3_path)
    song_name = track.split('.')
    #   转换格式
    sound = AudioSegment.from_file(mp3_path, format='mp3')
    vector_0 = np.zeros((len(info), 68))
    vector_1 = np.zeros((len(info), 68))

    for kk in range(len(info)):
        #   获取歌曲完整片段的特征
        wav_name = song_name[0] + str(kk) + '.wav'
        wav_path = os.path.join(tail, 'wavBlock', wav_name)
        #   截取完整片段
        blockData = sound[info[kk][0] * 1000:info[kk][1] * 1000]
        blockData.export(wav_path, format='wav')
        eigVector_0, eigVector_1 = Features(wav_path, 'second')
        print(kk)  # 标记程序运行进程
        #   得到一个片段的特征向量
        vector_0[kk, :] = np.mean(eigVector_0[0], 1)
        vector_1[kk, :] = np.mean(eigVector_1[0], 1)
    return vector_0, vector_1


def file_exists(file_path):
    if os.path.splitext(file_path) == '.mp3':
        if os.path.isfile(file_path):
            return file_path
        else:
            raise TypeError('文件不存在')
    else:
        raise TypeError('文件格式错误，后缀不为.mp3')


if __name__ == '__main__':

    # for path, dirs, files in os.walk('C:/Users/yoona/Desktop/music_test/'):
    #    for f in files:
    #        if not f.endwith('.mp3'):
    #            continue
    # 把路径组装到一起
    # path = r'C:\Users\yoona\Desktop\musictest'
    # f = 'CARTA - Aranya (Jungle Festival Anthem).mp3'
    # mp3_path = os.path.join(path, f)
    # =============================================================================
    # sa_b:a表示歌曲的序号，b表示歌曲的通道序号
    # =============================================================================
    # s1_0, s1_1, info1= compute_chunk_features(mp3_path)
    #    path_1 = file_exists(sys.argv[1])
    #    path_2 = file_exists(sys.argv[2])]

    path_1 = r'sample_audio\HMNSnor_Snoring child (ID 1070)_BSB.wav'
    path_2 = r'sample_audio\HMNSnor_Snoring child (ID 1070)_BSB.wav'

    s1_1, s1_2, info1 = compute_chunk_features(path_1)
    s2_1, s2_2, info2 = compute_chunk_features(path_2)

    sim_1 = similarity(s1_1, s2_1)  # 通道数1
    sim_2 = similarity(s1_2, s2_2)  # 通道数2

    info1_new = []
    info2_new = []

    for i, element in enumerate(sim_1):
        if element >= 0.5:
            info1_new.append(info1[i])

    if not info1_new:
        pos = np.argmax(sim_1)
        info1_new.append(info1[pos])
    s1_1, s1_2 = Compute_Bolck_Features(info1_new, path_1)

    for i, element in enumerate(sim_2):
        if element >= 0.5:
            info2_new.append(info2[i])

    if not info2_new:
        pos = np.argmax(sim_2)
        info2_new.append(info2[pos])
    s2_1, s2_2 = Compute_Bolck_Features(info2_new, path_2)

    sim_1 = similarity(s1_1, s2_1)  # 通道数1
    sim_2 = similarity(s1_2, s2_2)  # 通道数2

