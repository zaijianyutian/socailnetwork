import pickle
import numpy as np
from math import sqrt
import pandas as pd
import pymysql
from tqdm import tqdm
import sys
import random


def UserCorr(u_i_matrix, label):
    [m, n] = np.shape(u_i_matrix)
    matrix=np.copy(u_i_matrix)
    corr = np.zeros((m, m))
    avg = np.zeros(m)
    for i in range(m):  # 先计算每个用户的平均给分，作为中心化依据
        count = 0
        for j in range(n):
            if not label[i][j] == 1:  # 只考虑训练集数据
                continue
            avg[i] += matrix[i][j]
            count += 1
        if count == 0:
            avg[i] = 3  # 若用户在训练集中没有记录，则只能赋值3
        else:
            avg[i] /= count
    for i in tqdm(range(m)):
        for j in range(i + 1, m):
            count = 0
            xysum = 0
            x2sum = 0
            y2sum = 0
            for k in range(n):
                if not (label[i][k] == 1 and label[j][k] == 1):  # 必须要两者训练集中都有记录，才能计入
                    continue
                count += 1
                xysum += (matrix[i][k] - avg[i]) * (matrix[j][k] - avg[j])
                x2sum += (matrix[i][k] - avg[i]) ** 2
                y2sum += (matrix[j][k] - avg[j]) ** 2
            if count == 0:  # 训练集中两位用户看过的电影没有交集，则相似度只能设成0
                corr[i][j] = 0
                corr[j][i] = 0
            else:
                corr[i][j] = xysum / (sqrt(x2sum * y2sum) + 1e-6)  # 皮尔逊相似度计算
                corr[j][i] = corr[i][j]
    return corr,avg


def UserCF(matrix, corr, label, avg):
    [m, n] = np.shape(matrix)
    target = np.copy(matrix)
    RMSE = 0
    MAP = 0
    ecount = 0
    for i in tqdm(range(m)):
        for j in range(n):
            if label[i][j] == 1:  # 跳过训练集中已有记录
                continue
            upper = 0
            lower = 0
            count = 0
            for o in range(m):  # 从看过该电影的所有用户加权平均得到预测得分
                if not label[o][j] == 1:  # 前提是看过该电影
                    continue
                count += 1
                upper += corr[i][o] * (matrix[o][j] - avg[o])
                lower += abs(corr[i][o])
            if not (count == 0 or lower == 0):
                target[i][j] = avg[i] + upper / lower
            else:
                target[i][j] = avg[i]  # 若没有看过该电影的用户，只能赋值平均分
            if label[i][j] == 2:  # 与测试集数据进行对比
                RMSE += (target[i][j] - matrix[i][j]) ** 2
                MAP += abs(target[i][j] - matrix[i][j])
                ecount += 1
    if ecount:
        RMSE /= ecount
        MAP /= ecount
    return target, RMSE, MAP


def ItemCorr(u_i_matrix, label):
    [m, n] = np.shape(u_i_matrix)
    matrix = np.copy(u_i_matrix)
    corr = np.zeros((n, n))
    avg=np.zeros(n)
    for i in range(n):  # 先计算每部电影的平均得分，作为中心化依据
        count = 0
        for j in range(m):
            if not label[j][i] == 1:  # 只考虑训练集数据
                continue
            avg[i] += matrix[j][i]
            count += 1
        if count == 0:
            avg[i] = 3  # 若电影在训练集中没人评论，则只能赋值3
        else:
            avg[i] /= count
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            count = 0
            xysum = 0
            x2sum = 0
            y2sum = 0
            for k in range(m):
                if not (label[k][i] == 1 and label[k][j] == 1):  # 必须要两者训练集中都有记录，才能计入
                    continue
                count += 1
                xysum += (matrix[k][i] - avg[i]) * (matrix[k][j] - avg[j])
                x2sum += (matrix[k][i] - avg[i]) ** 2
                y2sum += (matrix[k][j] - avg[j]) ** 2
            if count == 0:  # 训练集中看过两部电影的观众没有交集，则相似度只能设成0
                corr[i][j] = 0
                corr[j][i] = 0
            else:
                corr[i][j] = xysum / (sqrt(x2sum * y2sum) + 1e-6)  # 皮尔逊相似度计算
                corr[j][i] = corr[i][j]
    return corr,avg


def ItemCF(matrix, corr, label, avg):
    [m, n] = np.shape(matrix)
    target = np.copy(matrix)
    RMSE = 0
    MAP = 0
    ecount = 0
    for i in tqdm(range(m)):
        for j in range(n):
            if label[i][j] == 1:  # 跳过训练集中已有记录
                continue
            upper = 0
            lower = 0
            count = 0
            for o in range(n):  # 从该用户看过的每一部电影进行加权平均
                if not label[i][o] == 1:  # 前提是看过该电影
                    continue
                count += 1
                upper += corr[j][o] * (matrix[i][o]-avg[o])
                lower += abs(corr[j][o])
            if not (count == 0 or lower == 0):
                target[i][j] = avg[j] + upper / lower
            else:
                target[i][j] = 3  # 若训练集中不存在该用户看过的电影，只能赋值3
            if label[i][j] == 2:  # 与测试集数据进行对比
                RMSE += (target[i][j] - matrix[i][j]) ** 2
                MAP += abs(target[i][j] - matrix[i][j])
                ecount += 1
    if ecount:
        RMSE /= ecount
        MAP /= ecount
    return target, RMSE, MAP


def recommend(uid, hist, pred, k=5, eps=0):
    recommend_list = []
    personal_score = np.copy(pred[uid])
    for i, watched in enumerate(hist[uid]):  # 将已经看过的电影排除
        if watched:
            personal_score[i] = 0
    personal_sort = np.argsort(personal_score)  # 从上至下排序
    general_score = np.average(pred, axis=0)
    general_sort = np.argsort(general_score)  # 以一定概率推荐总分较高电影，以应对信息茧房效应
    personal_iter = 0
    general_iter = 0
    for i in range(k):
        if np.random.rand() < eps:
            while general_sort[general_iter] in recommend_list:
                general_iter += 1
            recommend_list.append(general_sort[general_iter])
            general_iter += 1
        else:
            while personal_sort[personal_iter] in recommend_list:
                personal_iter += 1
            recommend_list.append(personal_sort[personal_iter])
            personal_iter += 1
    return recommend_list


if __name__ == '__main__':

    # userItem = np.load("item_user.npy")

    # thisLabel = np.load("label_item_full.npy")
    #
    # file=open("user_index_item_based.pkl","rb")
    # userID=pickle.load(file)
    # file.close()
    #
    file=open("item_index_item_based.pkl","rb")
    itemID=pickle.load(file)
    file.close()
    #
    # item_correlation, item_avg = ItemCorr(userItem, thisLabel)
    # np.save('itemCorr_full.npy', item_correlation)
    # user_correlation=np.load("itemCorr.npy")
    #
    # iPred,iRMSE,iMAP=ItemCF(userItem,item_correlation,thisLabel,item_avg)
    # np.save("itemPred_full.npy", iPred)



