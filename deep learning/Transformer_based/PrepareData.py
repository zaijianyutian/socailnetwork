import os
import math
import torch
import heapq
import random
import pandas as pd
import jieba
import datetime
#%%
class CorpusBag:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']#初始化保留关键字
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.hist_dict = {}#用于统计每个词语的出现次数

    def add_word(self, w):
        #向语料库中添加单词
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.hist_dict[w] = 1
        else:
            self. hist_dict[w] += 1

    def add_sentence(self, sentence):
        #分割句子，向语料库中添加单词
        for w in jieba.cut(sentence, cut_all=False):
            self.add_word(w)

    def prune(self, max_size=20000):
        '''
        剪枝，当语料库内容过多，修去语料库中频数较小的部分
        :param max_size: the max length of the corpus
        '''
        if len(self.hist_dict) > max_size:
            frequent_words = heapq.nlargest(max_size, self.hist_dict, key=self.hist_dict.get)#这里要改
            self.idx2word = self.idx2word[:4] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)




#%%
class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)

#%%
class DataLoader:
    def __init__(self,data_path, vocab_size):
        self.data_path = data_path
        self.vocab_size = vocab_size
        self.corpus = CorpusBag()
        self.user = EntityDictionary()
        self.item = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.corpus.prune(vocab_size)
        self.__unk = self.corpus.word2idx['<unk>']
        self.feature_set = set()#feature_set存储当前数据集出现过的feature
        self.train, self.valid, self.test = self.load_data(data_path)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pd.read_csv(data_path)
        for i in range(reviews.shape[0]):
            self.user.add_entity(reviews['USER_MD5'][i])
            self.item.add_entity(reviews['MOVIE_ID'][i])
            #加入tags
            tags = reviews['TAGS'][i].split('/')
            for j in range(3):
                self.corpus.add_word(tags[j])
            #加入genres
            genres = reviews['GENRES'][i].split('/')
            self.corpus.add_word(genres[0])
            #加入director
            director = reviews['DIRECTORS'][i].split('/')
            self.corpus.add_word(director[0])
            #加入year--need to consider again
            year = int(reviews['YEAR'][i])
            self.corpus.add_word(str(year))
            #加入评论
            comment = reviews['CONTENT'][i]
            self.corpus.add_sentence(comment)
            #处理rating
            rating = reviews['RATING'][i]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating








        # for review in reviews:
        #     self.user.add_entity(review['user'])
        #     self.item.add_entity(review['item'])
        #     (feature, opinion, sen, sentiment) = review['template']
        #     self.corpus.add_sentence(sen)
        #     self.corpus.add_word(feature)
        #     rating = review['rating']
        #     if self.max_rating < rating:
        #         self.max_rating = rating
        #     if self.min_rating > rating:
        #         self.min_rating = rating


    def load_data(self, data_path):
        data = []
        reviews = pd.read_csv(data_path)
        for i in range(reviews.shape[0]):
            feature = []
            # feature中加入tags
            tags = reviews['TAGS'][i].split('/')
            for j in range(3):
                feature.append(self.corpus.word2idx.get(tags[j],  self.corpus.word2idx['<unk>']))
                if tags[j] in self.corpus.word2idx:
                    #feature_set 的结构set，存储当前数据集出现过的feature
                    self.feature_set.add(tags[j])
                else:
                    self.feature_set.add('<unk>')
            # feature中加入genres
            genres = reviews['GENRES'][i].split('/')
            feature.append(self.corpus.word2idx.get(genres[0],  self.corpus.word2idx['<unk>']))
            if genres[0] in self.corpus.word2idx:
                # feature_set 的结构set，存储当前数据集出现过的feature
                self.feature_set.add(genres[0])
            else:
                self.feature_set.add('<unk>')
            # feature中加入director
            director = reviews['DIRECTORS'][i].split('/')
            feature.append(self.corpus.word2idx.get(director[0], self.corpus.word2idx['<unk>']))
            if director[0] in self.corpus.word2idx:
                # feature_set 的结构set，存储当前数据集出现过的feature
                self.feature_set.add(director[0])
            else:
                self.feature_set.add('<unk>')
            # 加入year--need to consider again
            year = int(reviews['YEAR'][i])
            feature.append(self.corpus.word2idx.get(str(year), self.corpus.word2idx['<unk>']))
            if str(year) in self.corpus.word2idx:
                # feature_set 的结构set，存储当前数据集出现过的feature
                self.feature_set.add(str(year))
            else:
                self.feature_set.add('<unk>')

            comment = reviews['CONTENT'][i]
            data.append({'user': self.user.entity2idx[reviews['USER_MD5'][i]],
                                          'item': self.item.entity2idx[reviews['MOVIE_ID'][i]],
                                          'rating': reviews['RATING'][i],
                                          'text': self.seq2ids(comment),
                                          'feature': feature})


        # for review in reviews:
        #     (feature, opinion, sen, sentiment) = review['template']
        #     data.append({'user': self.user.entity2idx[review['user']],
        #                  'item': self.item.entity2idx[review['item']],
        #                  'rating': review['rating'],
        #                  'text': self.seq2ids(sen),
        #                  'feature': self.corpus.word2idx.get(feature,  self.corpus.word2idx['<unk>'])})
        #     if feature in self.corpus.word2idx:
        #         #feature_set 的结构set，存储当前数据集出现过的feature
        #         self.feature_set.add(feature)
        #     else:
        #         self.feature_set.add('<unk>')

        #划分数据集，train:test:validation=8:1:1
        train_num = math.ceil(reviews.shape[0]*0.8)
        test_num = math.ceil(reviews.shape[0]*0.1)
        train_idx = set(random.sample(range(reviews.shape[0]),train_num))
        left_reviews = set(range(reviews.shape[0]))-train_idx
        test_idx = set(random.sample(list(left_reviews), test_num))
        vali_idx = left_reviews-test_idx
        train_idx = list(train_idx)
        test_idx = list(test_idx)
        vali_idx = list(vali_idx)
        #读入数据
        train, valid, test = [], [], []
        for idx in train_idx:
            train.append(data[idx])
        for idx in vali_idx:
            valid.append(data[idx])
        for idx in test_idx:
            test.append(data[idx])
        return train, valid, test

    def seq2ids(self, seq):
        return [self.corpus.word2idx.get(w, self.corpus.word2idx['<unk>']) for w in jieba.cut(seq, cut_all=False)]

#%%
class Batchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        user, item, rating, template, feature = self.initialize(data, word2idx, seq_len)
        self.user = torch.tensor(user, dtype=torch.int64).contiguous()
        self.item = torch.tensor(item, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(rating, dtype=torch.float).contiguous()
        self.seq = torch.tensor(template, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(feature, dtype=torch.int64).contiguous()
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(len(data)))
        self.total_step = int(math.ceil(len(data)/ self.batch_size))
        self.step = 0

    def initialize(self,data, word2idx, seq_len=15):
        '''
        初始化，读入数据，分配给user item feature
        :param data: 读入的数据
        '''
        user, item, rating, template, feature = [], [], [], [], []
        for example in data:
            user.append(example['user'])
            item.append(example['item'])
            rating.append(example['rating'])
            template.append(pack_sentence(example['text'], seq_len, pad=word2idx['<pad>'], bos=word2idx['<bos>'], eos=word2idx['<eos>']))
            feature.append(example['feature'])
        return user, item, rating, template, feature


    def next_batch(self,shuffle=False):
        '''
        读入下一个batch，可选择是否shuffle
        '''
        if self.step == self.total_step:
            #此条件判断是否已经读完，若已经读完，则从头再来
            self.step = 0
            if shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        end = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:end]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 6)
        return user, item, rating, seq, feature
#%%
def ids2tokens(ids, word2idx, idx2word):
    '''
    由id映射回word
    '''
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens
#%%
def pack_sentence(sentence, format_len, pad, bos, eos):
    '''
    包装句子，为其加上开始结束符，句子规范化，小于规定长度的，在后面添加padding，大于规定长度的，截断
    :param format_len:规定长度
    '''
    length = len(sentence)
    if length >= format_len:
        return [bos] + sentence[:format_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (format_len - length)


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '