from copy import deepcopy

import torch
from math import log2
#参数文件

config = {
    # movielens
    'num_rate': 6,
    'num_genre': 25,
    'num_director': 2186,
    'num_year': 81,
    'embedding_dim': 32,
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    # bookcrossing
    'num_year_bk': 80,
    'num_author': 25593,
    'num_publisher': 5254,
    'num_age_bk': 106,
    'num_location': 65,
    # amazon  'categories', 'title', 'price', 'sales_type', 'sales_rank', 'brand'
    'num_cate': 1318,
    'num_title': 484209,
    'num_price': 22166,
    'num_brand': 9993,
    'num_type':28,
    'num_rank':23234,  #!!!!!!!!!!!!!!!!!!!!

    # yelp
    ##user_id:token	user_review_count:float	yelping_since:float	user_useful:float	user_funny:float	user_cool:float	fans:float	average_stars:float	compliment_hot:float
    # compliment_more:float	compliment_profile:float	compliment_cute:float	compliment_list:float
    # compliment_note:float	compliment_plain:float	compliment_cool:float	compliment_funny:float	compliment_writer:float	compliment_photos:float
    'num_count':1877,
    'num_u_useful':5112,
    'num_u_funny':3730,
    'num_u_cool': 4314,
    'num_fans':659,
    'num_stars': 400,
    'num_c_hot': 1400,
    'num_c_more': 357,
    'num_c_profile':362,
    'num_c_cute':332,
    'num_c_list':196,
    'num_c_note':984,
    'num_c_plain':1612,
    'num_c_cool':1635,
    'num_c_funny': 1635,
    'num_c_writer': 861,
    'num_c_photos': 971,
## city:token_seq	state:token	postal_code:token	latitude:float	longitude:float	item_stars:float	item_review_count:float	 is_open:float	categories:token_seq
    'num_city':1250,
    'num_state':37,
    'num_postal_code':18604,
    'num_item_stars':9,
    'num_item_review_count':1320,
    'num_categories':1377,

    # cuda setting
    'use_cuda': False,#True
    # save_model setting
    'inner': 2,
    'global_lr': 1e-4,#gamma。global_lr用于更新meta-save_model
    'local_lr': 1e-5,#需要学习的参数
    'batch_size': 512,
    'num_epoch':100,
    'support_size':100,
    'query_size':20,
    'C':0.6,#threshold for user-specific save_model used to global updating
    'lamda_1':1e-7,#regularization
     'lamda_2':1e-7,
     #'alpha_hat':0.1,#learning_rate
    # candidate selection
    'num_candidate': 20,
    'cluster_n': 5
}

#n_y：y的取值个数
#u_in_dim是input_loading里embedding层数
default_info = {
    'movielens': {'n_y': 5, 'u_in_dim': 4, 'i_in_dim': 4},
    'yelp': {'n_y': 5, 'u_in_dim': 10, 'i_in_dim': 6},
    'bookcrossing': {'n_y': 10, 'u_in_dim': 2, 'i_in_dim': 3},
    'amazon': {'n_y': 5, 'i_in_dim': 4}  #user有几个属性就是几
}

#参数文件
states = ["warm_state", "user_cold_state"]#, "item_cold_state", "user_and_item_cold_state"]

def grads_sum(raw_grads_list, new_grads_list):
    return [raw_grads_list[i]+new_grads_list[i] for i in range(len(raw_grads_list))]


def update_parameters(params, grads, lr):
    return [params[i] - lr*grads[i] for i in range(len(params))]


# ===============================================
def activation_func(name):
    name = name.lower()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == 'softmax':
        return torch.nn.Softmax()
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(0.1)
    else:
        return torch.nn.Sequential()


# ===============================================
def mae(ground_truth, test_result):
    if len(ground_truth) > 0:
        pred_y = torch.argmax(test_result, dim=1)
        sub = ground_truth-pred_y
        abs_sub = torch.abs(sub)
        out = torch.mean(abs_sub.float(), dim=0)
    else:
        out = 1
    return out


def ndcg(ground_truth, test_result, top_k=3):
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

def get_params(param_list):
    params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.data)
            params.append(value)
            del value
        count += 1
    return params


def get_zeros_like_params(param_list):
    zeros_like_params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(torch.zeros_like(param.data))
            zeros_like_params.append(value)
        count += 1
    return zeros_like_params

def init_params(param_list, init_values):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count])
            init_count += 1
        count += 1
def get_grad(param_list):
    count = 0
    param_grads = []
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.grad)
            param_grads.append(value)
            del value
        count += 1
    return param_grads


def binary_conversion(var: int):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    assert isinstance(var, int)
    #if var <= 1024:
    return round(var / 1024, 2)
    # else:
    #     return round(var / (1024 ** 2), 2)#f'占用 {round(var / (1024 ** 2), 2)} MB内存'


def mae(ground_truth, test_result):
    if len(ground_truth) > 0:
        pred_y = torch.argmax(test_result, dim=1)  # 分类问题的解
        # pred_y = test_result.view(-1)
        abs_sub = torch.abs(ground_truth - pred_y)
        out = torch.mean(abs_sub.float(), dim=0)
    else:
        out = 1
    return out

def ndcg(ground_truth, test_result, top_k=3):
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
