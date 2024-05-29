from CF import ItemCorr, recommend
import pickle
import numpy as np
from math import sqrt
import pandas as pd
from tqdm import tqdm
import random


def ICFP(matrix, corr, user_image, dir, act, gen, label, avg):
    [m, n] = np.shape(matrix)
    target = np.copy(matrix)
    RMSE = 0
    MAE = 0
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
                if not sum(user_image[i]) == 0:  # 自适应地为不同属性的相似度赋权重，组合最终相似度
                    totCorr = 0.5 * corr[j][o] + 0.5 * (
                                user_image[i][0] * dir[j][o] + user_image[i][1] * act[j][o] + user_image[i][2] * gen[j][
                            o]) / sum(user_image[i])
                else:
                    totCorr = corr[j][o]
                upper += totCorr * (matrix[i][o] - avg[o])
                lower += abs(totCorr)
            if not (count == 0 or lower == 0):
                target[i][j] = avg[j] + upper / lower
            else:
                target[i][j] = 3  # 若训练集中不存在该用户看过的电影，只能赋值3
            if label[i][j] == 2:  # 与测试集数据进行对比
                RMSE += (target[i][j] - matrix[i][j]) ** 2
                MAE += abs(target[i][j] - matrix[i][j])
                ecount += 1
    if ecount:
        RMSE /= ecount
        MAE /= ecount
    return target, RMSE, MAE


if __name__ == '__main__':
    """
    统计各属性各实体对应电影数量，并选出几个重要的实体。共选出50位导演，50位演员和26个类别
    """
    # movies = pd.read_csv("movie_selected_1k.csv")
    # directors = []
    # dir_id = {}
    # dk = 0
    # actors = []
    # act_id = {}
    # ak = 0
    # genres = []
    # gen_id = {}
    # gk = 0
    # for i in range(len(movies)):
    #     dirs = movies["DIRECTORS"][i].split("/")
    #     acts = movies["ACTORS"][i].split("/")
    #     gens = movies["GENRES"][i].split("/")
    #     for dir in dirs:
    #         if dir not in dir_id:
    #             directors.append((0, dir))
    #             dir_id[dir] = dk
    #             dk += 1
    #         directors[dir_id[dir]] = (directors[dir_id[dir]][0] + 1, dir)
    #     count = 0
    #     for act in acts:  # 演员只取前5名
    #         if count >= 5:
    #             break
    #         count += 1
    #         if act not in act_id:
    #             actors.append((0, act))
    #             act_id[act] = ak
    #             ak += 1
    #         actors[act_id[act]] = (actors[act_id[act]][0] + 1, act)
    #     for gen in gens:
    #         if gen == "剧情":  # 忽略一个几乎无区分度的分类
    #             continue
    #         if gen not in gen_id:
    #             genres.append((0, gen))
    #             gen_id[gen] = gk
    #             gk += 1
    #         genres[gen_id[gen]] = (genres[gen_id[gen]][0] + 1, gen)
    # directors.sort(reverse=True)
    # dir_sel = []
    # if len(directors) > 50:
    #     for j in range(50):
    #         if directors[j][0] < 2:
    #             break
    #         dir_sel.append(directors[j][1])
    # else:
    #     for j in directors:
    #         dir_sel.append(j[1])
    # actors.sort(reverse=True)
    # act_sel = []
    # if len(actors) > 50:
    #     for j in range(50):
    #         if actors[j][0] < 2:
    #             break
    #         act_sel.append(actors[j][1])
    # else:
    #     for j in actors:
    #         act_sel.append(j[1])
    # genres.sort(reverse=True)
    # gen_sel = []
    # if len(genres) > 50:
    #     for i in range(50):
    #         if genres[i][0] < 2:
    #             break
    #         gen_sel.append(genres[i][1])
    # else:
    #     for j in genres:
    #         gen_sel.append(j[1])
    # print(gen_sel)
    # print(act_sel)
    # print(dir_sel)
    # file = open("genres.pkl", "wb")
    # pickle.dump(gen_sel, file)
    # file.close()
    # file = open("directors.pkl", "wb")
    # pickle.dump(dir_sel, file)
    # file.close()
    # file = open("actors.pkl", "wb")
    # pickle.dump(act_sel, file)
    # file.close()

    """
    统计每个用户对每个标签的喜爱程度（公式：平均给分+5*sqrt（频率））
    """
    # userItem = np.load("item_user.npy")
    # file = open("item_index_item_based.pkl", "rb")
    # itemIndex = pickle.load(file)
    # file.close()
    # label = np.load("label_item_full.npy")
    #
    # [m, n] = np.shape(userItem)
    # user_image = np.zeros((m, 50 + 50 + 26))
    # for i in tqdm(range(m)):
    #     avg = np.zeros(50 + 50 + 26)
    #     freq = np.zeros(50 + 50 + 26)
    #     count=0
    #     for j in range(n):
    #         if label[i][j] == 1:
    #             count+=1
    #             movie = movies[movies["MOVIE_ID"] == itemIndex[j]]
    #             for dir in movie["DIRECTORS"].values[0].split("/"):
    #                 if dir in dir_sel:
    #                     avg[dir_sel.index(dir)] += userItem[i][j]
    #                     freq[dir_sel.index(dir)] += 1
    #             for act in movie["ACTORS"].values[0].split("/"):
    #                 if act in act_sel:
    #                     avg[act_sel.index(act)+50] += userItem[i][j]
    #                     freq[act_sel.index(act)+50] += 1
    #             for gen in movie["GENRES"].values[0].split("/"):
    #                 if gen in gen_sel:
    #                     avg[gen_sel.index(gen)+100] += userItem[i][j]
    #                     freq[gen_sel.index(gen)+100] += 1
    #     for k in range(50+50+26):
    #         if freq[k]:
    #             avg[k]/=freq[k]
    #         else:
    #             avg[k]=3
    #     if count:
    #         freq=freq/count
    #     for k in range(50+50+26):
    #         user_image[i][k]=avg[k]+5*sqrt(freq[k])
    # np.save("user_image.npy",user_image)

    """
    计算每个用户每种属性的类间方差
    """
    # user_image=np.load("user_image.npy")
    # userItem=np.load("item_user.npy")
    # [m,n]=np.shape(userItem)
    # bet_var=np.zeros((m,3))
    # for i in range(m):
    #     bet_var[i][0]=np.var(user_image[i][0:50])
    #     bet_var[i][1]=np.var(user_image[i][50:100])
    #     bet_var[i][2]=np.var(user_image[i][100:126])
    # u=random.randint(0, 800)
    # print(bet_var[u])
    # print(user_image[u])
    # print(userItem[u])
    # np.save("user_var.npy",bet_var)

    """
    得到物品在每个属性上的相似度
    """
    # file = open("genres.pkl", "rb")
    # gen_sel=pickle.load(file)
    # file.close()
    # file = open("directors.pkl", "rb")
    # dir_sel=pickle.load(file)
    # file.close()
    # file = open("actors.pkl", "rb")
    # act_sel=pickle.load(file)
    # file.close()
    # movies=pd.read_csv("movie_selected_1k.csv")
    # file=open("item_index_item_based.pkl","rb")
    # itemIndex=pickle.load(file)
    # file.close()
    # n=len(itemIndex)
    # DirCorr=np.zeros((n,n))
    # ActCorr=np.zeros((n,n))
    # GenCorr=np.zeros((n,n))
    # for i in range(n):
    #     movie1=movies[movies["MOVIE_ID"]==itemIndex[i]]
    #     temp=movie1["DIRECTORS"].values[0].split("/")
    #     dirs1 = [t for t in dir_sel if t in temp]
    #     temp=movie1["ACTORS"].values[0].split("/")
    #     acts1 = [t for t in act_sel if t in temp]
    #     temp=movie1["GENRES"].values[0].split("/")
    #     gens1 = [t for t in gen_sel if t in temp]
    #     for j in range(i+1,n):
    #         movie2 = movies[movies["MOVIE_ID"] == itemIndex[i]]
    #         temp = movie2["DIRECTORS"].values[0].split("/")
    #         dirs2 = [t for t in dir_sel if t in temp]
    #         temp = movie2["ACTORS"].values[0].split("/")
    #         acts2 = [t for t in act_sel if t in temp]
    #         temp = movie2["GENRES"].values[0].split("/")
    #         gens2 = [t for t in gen_sel if t in temp]
    #         intersection_d=[t for t in dirs2 if t in dirs1]
    #         if not len(intersection_d)==0:
    #             DirCorr[i][j] = 2 * len(intersection_d) / (len(dirs1) + len(dirs2))
    #             DirCorr[j][i] = DirCorr[i][j]
    #         intersection_a = [t for t in acts2 if t in acts1]
    #         if not len(intersection_a)==0:
    #             ActCorr[i][j] = 2 * len(intersection_a) / (len(acts1) + len(acts2))
    #             ActCorr[j][i]=ActCorr[i][j]
    #         intersection_g = [t for t in gens2 if t in gens1]
    #         if not len(intersection_g)==0:
    #             GenCorr[i][j] = len(intersection_g) / (len(gens1) + len(gens2))
    #             GenCorr[j][i] = GenCorr[i][j]
    # GenCorr=GenCorr-np.mean(GenCorr)  # 中心化
    # ActCorr=ActCorr-np.mean(ActCorr)
    # DirCorr=DirCorr-np.mean(DirCorr)
    # np.save("dir_corr.npy",DirCorr)
    # np.save("act_corr.npy",ActCorr)
    # np.save("gen_corr.npy",GenCorr)

    """
    进行预测
    """
    # userItem=np.load("item_user.npy")
    # itemCorr=np.load("itemCorr_full.npy")
    # dirCorr=np.load("dir_corr.npy")
    # actCorr=np.load("act_corr.npy")
    # genCorr=np.load("gen_corr.npy")
    # userVar=np.load("user_var.npy")
    # thisLabel=np.load("label_item_full.npy")
    # [m,n]=np.shape(userItem)
    # avg=np.zeros(n)
    # for i in range(n):  # 先计算每部电影的平均得分，作为中心化依据
    #     count = 0
    #     for j in range(m):
    #         if not thisLabel[j][i] == 1:  # 只考虑训练集数据
    #             continue
    #         avg[i] += userItem[j][i]
    #         count += 1
    #     if count == 0:
    #         avg[i] = 3  # 若电影在训练集中没人评论，则只能赋值3
    #     else:
    #         avg[i] /= count
    # pPred,_,_=ICFP(userItem,itemCorr,userVar,dirCorr,actCorr,genCorr,thisLabel,avg)
    # np.save("item_plus_predict.npy",pPred)
    userItem = np.load("item_user.npy")
    m, n = np.shape(userItem)
    pPred = np.load("item_plus_predict.npy")
    iPred = np.load("itemPred_full.npy")
    u = random.randint(0, m)
    print(u)
    print(recommend(u, userItem, pPred))
    print(recommend(u, userItem, iPred))
