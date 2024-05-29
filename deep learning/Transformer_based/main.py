import os
import math
import torch
import argparse
import torch.nn as nn
from module import PETER
from PrepareData import DataLoader, Batchify, ids2tokens, now_time
from metrics import  rouge_score, bleu_score, RMSE, MAE, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity,unique_sentence_percent


# ----------------以下设置超参数-----------------

#超参数写法参照了1https://github.com/lileipisces/PETER，
parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the csv data')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of embeddings')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./peter/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--rating_reg', type=float, default=0.1,
                    help='regularization on recommendation task')
parser.add_argument('--context_reg', type=float, default=1.0,
                    help='regularization on context prediction task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
parser.add_argument('--peter_mask', action='store_true',
                    help='True to use peter mask; Otherwise left-to-right mask')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
args = parser.parse_args()



# 设置随机种子
torch.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')   #设置训练设备，优先GPU
#指定checkpoint文件夹
if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)


#---------------------------以下加载数据，准备模型-----------------------------
data_all = DataLoader(args.data_path, args.vocab_size)
word2idx = data_all.corpus.word2idx
idx2word = data_all.corpus.idx2word
feature_set = data_all.feature_set
train_data = Batchify(data_all.train, word2idx, args.words, args.batch_size, shuffle=True)
val_data = Batchify(data_all.valid, word2idx, args.words, args.batch_size)
test_data = Batchify(data_all.test, word2idx, args.words, args.batch_size)


src_len = 2 + train_data.feature.size(1)  # [user, item, feature] 长度

tgt_len = args.words + 1  # added <bos> or <eos>
ntokens, nuser, nitem= len(data_all.corpus), len(data_all.corpus), len(data_all.corpus)
pad_idx = word2idx['<pad>']
#准备模型
model = PETER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)



#-------------------------以下 train, evaluate, generate函数-------------------------



def train(data):

    model.train()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, explan, feature = data.next_batch()
        batch_size = user.size(0)
        user = user.to(device)
        item = item.to(device)
        rating = rating.to(device)
        explan = explan.t().to(device)
        feature = feature.t().to(device)  # (6, batch_size)
        #将feature和explanation拼接成text输入
        text = torch.cat([feature, explan[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)


        optimizer.zero_grad()
        #使用三种loss：context_loss, text_loss, rating_loss
        log_word_prob, log_context_dis, rating_p = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))
        #分别计算3种loss
        c_loss = text_criterion(context_dis.view(-1, ntokens), explan[1:-1].reshape((-1,)))
        r_loss = rating_criterion(rating_p, rating)
        t_loss = text_criterion(log_word_prob.view(-1, ntokens), explan[1:].reshape((-1,)))
        loss = args.text_reg * t_loss + args.context_reg * c_loss + args.rating_reg * r_loss
        loss.backward()

        # 训练的时候发现问题， 这里参照了原文的训练方法，`clip_grad_norm` 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        context_loss += batch_size * c_loss.item()
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        # 每200step进行一次报告
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_c_loss = context_loss / total_sample
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'context loss {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_c_loss), math.exp(cur_t_loss), cur_r_loss, data.step, data.total_step))
            context_loss = 0.
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):

    model.eval()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, explan, feature = data.next_batch()
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            explan = explan.t().to(device)  # (tgt_len + 1, batch_size)
            feature = feature.t().to(device)  # (1, batch_size)
            # 将feature和explanation拼接成text输入
            text = torch.cat([feature, explan[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            #和train一样使用三种loss：context_loss, text_loss, rating_loss
            log_word_prob, log_context_dis, rating_p = model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))
            #分别计算三种loss
            c_loss = text_criterion(context_dis.view(-1, ntokens), explan[1:-1].reshape((-1,)))
            r_loss = rating_criterion(rating_p, rating)
            t_loss = text_criterion(log_word_prob.view(-1, ntokens), explan[1:].reshape((-1,)))

            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, explan, feature = data.next_batch()
            user = user.to(device)  #
            item = item.to(device)
            bos = explan[:, 0].unsqueeze(0).to(device)  # (1, batch_size)
            feature = feature.t().to(device)
            #生成截断无需将explanation输入，而是输入bos开始自回归生成解释
            text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)

            start_idx = text.size(0)
            for idx in range(args.words):
                #自回归，一次生成一个单词
                if idx == 0:
                    log_word_prob, log_context_dis, rating_p = model(user, item, text, False)  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_predict.extend(rating_p.tolist())
                    context = predict(log_context_dis, topk=args.words)  # (batch_size, words) 执行context prediction 用1处的编码加入整体的生成
                    context_predict.extend(context.tolist())
                else:
                    log_word_prob, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)
                word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # 每次生成概率最大的词
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)
            ids = text[start_idx:].t().tolist()
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break

    # 各metric与原文一样，方便对比
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    rmse = RMSE(predicted_rating, data_all.max_rating, data_all.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(rmse))
    mae = MAE(predicted_rating, data_all.max_rating, data_all.min_rating)
    print(now_time() + 'MAE {:7.4f}'.format(mae))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    # for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
    #     text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    for (u, i, real, fake, r_real, r_pred) in zip(test_data.user.tolist(), test_data.item.tolist(), text_test,
                                                  text_predict, test_data.rating.tolist(), rating_predict):
        text_out += '用户：{}\t电影：{}\n 原评论：{}\n生成的解释：{}\n真实评分：{}\t预测评分{}\n\n'.format(data_all.user.idx2entity[u],
                                                                                   data_all.item.idx2entity[i], real,
                                                                                   fake, r_real, r_pred)
    return text_out


def predict(log_context_dis, topk):
    '''
    预测topk 个单词
    :param log_context_dis:log likelyhood of each word
    :param topk:指定前k个词
    '''
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context



#----------------以下进行模型训练与评估--------------

#模型训练
print('-' * 30+'以下是训练过程'+'-' * 30)
min_loss = float('inf')     #记录最小损失
endure = 0                  #endure次数记录
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_c_loss, val_t_loss, val_r_loss = evaluate(val_data)
    #validation loss 为 val_t_loss + val_r_loss
    val_loss = val_t_loss + val_r_loss
    print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_loss))
    # 当validation loss小于min_loss 即该步训练有效， 保存模型
    if val_loss < min_loss:
        bmin_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure += 1
        print(now_time() + 'Endured {} time(s)'.format(endure))
        if endure== args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
        # validation loss没有进步， 调整学习率
        scheduler.step()
        print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

#模型评估
#载入最终训练好的模型
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

print('-' * 30+'以下是在测试集上的表现'+'-' * 30)
test_c_loss, test_t_loss, test_r_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
    math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))

print(now_time() + 'Generating text')
text_o = generate(test_data)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))