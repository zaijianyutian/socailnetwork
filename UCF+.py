from CF import UserCorr, recommend
import pickle
import numpy as np
from math import sqrt
import pandas as pd
from tqdm import tqdm
import random
import networkx as nx
import heapq


def UCFP(matrix, corr, social_network, social_sim, user_cen, label, avg):
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
                sim = corr[i][o] + social_sim[i][o]
                if social_network[i][o] and user_cen[o]:  # 若是关注的高中心度用户，权重扩大至3倍
                    sim *= 3
                upper += sim * (matrix[o][j] - avg[o])
                lower += abs(sim)
            if not (count <= 1 or lower == 0):  # 看过的人必须超过1个，如果不进行限制，推荐中可能会出现噪音。
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


def SocialCorr(social_network):
    m = len(social_network)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(m))
    for i in range(m):
        for j in range(m):
            if social_network[i][j]:
                graph.add_edge(i, j)
    centrality = nx.in_degree_centrality(graph)  # 使用入度中心度计算
    cen = np.zeros(m)
    is_cen = np.zeros(m)
    for i in centrality:
        cen[i] = centrality[i]
    threshold = heapq.nlargest(50, cen)[-1]  # 选择中心度前50的节点作为“资深影评人”
    for i in range(m):
        if cen[i] >= threshold:
            is_cen[i] = 1
    social_sim=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            ni=[u for u in graph[i] if is_cen[u]]  # 在重合的邻居中，只考虑资深影评人
            nj=[u for u in graph[j] if is_cen[u]]
            intersection=set(ni).intersection(nj)
            union=set(ni).union(nj)
            if len(union):
                social_sim[i][j]=len(intersection)/len(union)  # Jaccard相似度计算（只考虑资深影评人）
                social_sim[j][i]=social_sim[i][j]
    return social_sim,is_cen






if __name__ == '__main__':
    userItem=np.load("user_item.npy")
    [m,n]=np.shape(userItem)
    social_network=np.load("social_network.npy")
    social_sim,is_cen=SocialCorr(social_network)
    np.save("social_sim.npy",social_sim)
    np.save("user_cen.npy",is_cen)
    thisLabel=np.load("label_full.npy")
    userCorr=np.load("userCorr_full.npy")
    avg=np.zeros(m)
    for i in range(m):  # 先计算每个用户的平均给分，作为中心化依据
        count = 0
        for j in range(n):
            if not thisLabel[i][j] == 1:  # 只考虑训练集数据
                continue
            avg[i] += userItem[i][j]
            count += 1
        if count == 0:
            avg[i] = 3  # 若用户在训练集中没有记录，则只能赋值3
        else:
            avg[i] /= count
    pPred,_,_=UCFP(userItem,userCorr,social_network,social_sim,is_cen,thisLabel,avg)
    np.save("user_plus_predict.npy",pPred)
    u=random.randint(0,m)
    print(u,recommend(u,userItem,pPred))
