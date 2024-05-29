import math
from rouge import rouge
from bleu import compute_bleu

def MAE(predicted, max_rating, min_rating):
    '''
    计算MAE， 对predicted_rating-label_rating绝对值求和
    '''
    mae = 0
    for pred, lab in predicted:
        if pred > max_rating:
             pred = max_rating
        if pred < min_rating:
            pred = min_rating
        mae += abs(pred - lab)
    mae = mae/len(predicted)
    return mae

def RMSE(predicted, max_rating, min_rating):
    '''
    计算RMSE， 对predicted_rating-label_rating的平方值求和
    '''
    rmse = 0
    for pred, lab in predicted:
        if pred > max_rating:
            pred = max_rating
        if pred < min_rating:
            pred = min_rating
        rmse += (pred - lab)**2
    rmse = rmse / len(predicted)
    return rmse

def rouge_score(references, generated):
    '''
    参照论文《ROUGE: Recall-oriented understudy for gisting evaluation》，
    从https://github.com/tensorflow/nmt/blob/master/nmt/scripts/rouge.py掉包
    Args:
        references:监督用句子
        generated:生成的句子

    '''
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    '''
    计算bleu值，从https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py掉包
    Args:
        references:监督用句子
        generated:生成的句子
        n_gram:词对长度
    '''
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100

def feature_detect(seq_batch, feature_set):
    '''
    返回featrue 的集合，从seq_batch中
    Args:
        seq_batch:
        feature_set:

    Returns:

    '''
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator

def two_seq_same(sa, sb):
    '''
    判断两个句子是否相同
    '''
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    '''
    计算sequence_batch中unique_sentence的比例
    '''
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)