
import torch
from math import log2

#ground_truth:实际分数  test_result: 模型输出分数
def ndcg(ground_truth, test_result, top_k=5):
    pred_y = torch.argmax(test_result, dim=1)
    sort_real_y, sort_real_y_index = ground_truth.clone().detach().sort(descending=True)
    sort_pred_y, sort_pred_y_index = pred_y.clone().detach().sort(descending=True)
    pred_sort_y = ground_truth[sort_pred_y_index][:top_k]
    top_pred_y, _ = pred_sort_y.sort(descending=True)

    ideal_dcg = 0
    n = 1
    for value in sort_real_y[:top_k]:
        i_dcg = (2**float(value+1) - 1)/log2(n+1)
        ideal_dcg += i_dcg
        n += 1

    pred_dcg = 0
    n = 1
    for value in top_pred_y:
        p_dcg = (2**float(value+1) - 1)/log2(n+1)
        pred_dcg += p_dcg
        n += 1

    n_dcg = pred_dcg/ideal_dcg
    return n_dcg


def hit_num(ground_truth, test_result, top_k=5):
    pass