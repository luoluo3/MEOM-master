import os
from collections import OrderedDict

import torch
import pickle
import random

from tqdm import tqdm

from options import config
from math import log2
from torch.utils.tensorboard import SummaryWriter

'''
    :param ground_truth: real scores
    :param test_result: predicted scores
    only need y not x
    '''


# test_result应该是给用户模型所有的item,然后计算的
def ndcg(ground_truth, test_result, topk=[1, 3]):
    # pred_y = torch.argmax(test_result, dim=1)
    pred_y = test_result.view(-1)
    sort_real_y, sort_real_y_index = ground_truth.clone().detach().sort(descending=True)
    sort_pred_y, sort_pred_y_index = pred_y.clone().detach().sort(descending=True)
    ndcg = []
    hr = []
    for top_k in topk:
        pred_sort_y = ground_truth[sort_pred_y_index][:top_k]
        top_pred_y, _ = pred_sort_y.sort(descending=True)

        ideal_dcg = 0
        n = 1
        for value in sort_real_y[:top_k]:
            i_dcg = (2 ** float(value + 1) - 1) / log2(n + 1)
            ideal_dcg += i_dcg
            n += 1

        pred_dcg = 0
        n = 1
        for value in top_pred_y:
            p_dcg = (2 ** float(value + 1) - 1) / log2(n + 1)
            pred_dcg += p_dcg
            n += 1

        ndcg.append(pred_dcg / ideal_dcg)

        # hits=0
        # for i in sort_pred_y_index[:top_k]:
        #     if i in sort_real_y_index[:top_k]:
        #         hits+=1
        # hr.append(hits/top_k)
    return ndcg  # ,hr


def mae(ground_truth, test_result):
    if len(ground_truth) > 0:
        # pred_y = torch.argmax(test_result, dim=1)  # 分类问题的解
        pred_y = test_result.view(-1)
        abs_sub = torch.abs(ground_truth - pred_y)
        out = torch.mean(abs_sub.float(), dim=0)
    else:
        out = 1
    return out

# def ndcg(ground_truth, test_result, top_ks=[1,3]):
#     pred_y = torch.argmax(test_result, dim=1)
#     sort_real_y, sort_real_y_index = ground_truth.clone().detach().sort(descending=True)
#     sort_pred_y, sort_pred_y_index = pred_y.clone().detach().sort(descending=True)
#     ndcg=[]
#     for top_k in top_ks:
#         pred_sort_y = ground_truth[sort_pred_y_index][:top_k]
#         top_pred_y, _ = pred_sort_y.sort(descending=True)
#
#         ideal_dcg = 0
#         n = 1
#         for value in sort_real_y[:top_k]:
#             i_dcg = (2**float(value+1) - 1)/log2(n+1)
#             ideal_dcg += i_dcg
#             n += 1
#
#         pred_dcg = 0
#         n = 1
#         for value in top_pred_y:
#             p_dcg = (2**float(value+1) - 1)/log2(n+1)
#             pred_dcg += p_dcg
#             n += 1
#
#         n_dcg = pred_dcg/ideal_dcg
#         ndcg.append(n_dcg)
#     return ndcg

def get_current_profile(u_dict, u_ids):
    current_profile = OrderedDict()
    for id in u_ids:
        current_profile[id] = u_dict[id]
    return current_profile


def training(model, cluster, users_profile, current_dataset, test_dataset, epoch_num, batch_size, stage_id, model_save=True,
             model_filename=None, reg_name=None, users_id=None, cus_id=None, rep_name=None, dynamic_center= None, dynamic_data=None):
    TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_log')
    if config['use_cuda']:
        model.cuda()

    '''get current user cluster based on user's profile'''
    # cluster(get_current_profile(users_profile, users_id), config['cluster_n'])

    training_set_size = len(current_dataset)
    print("*****************training_set_size", training_set_size)
    # batch_size = int(training_set_size / batch_num)
    batch_num = round(training_set_size / batch_size)
    print("*****************batch_num", batch_num)
    print("*****************batch_size", batch_size)
    model.train()

    tr_x, tr_y, te_x, te_y = zip(*test_dataset)
    sx_t, sy_t, qx_t, qy_t = zip(*current_dataset)


    epoch_ndcg_1, epoch_ndcg_3, c_epoch_ndcg_1, c_epoch_ndcg_3 = [], [], [], []
    epoch_mae, c_epoch_mae = [], []
    # epoch_hr_5,c_epoch_hr_5,epoch_hr_10,c_epoch_hr_10=[],[],[],[]
    for en in range(epoch_num):
        # random.shuffle(current_dataset)#打乱列表
        b_ndcg_1,  b_ndcg_3,  = [], []
        b_mae = []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(sx_t[batch_size * i:batch_size * (i + 1)])       # 取batch_size个用户训练
                supp_ys = list(sy_t[batch_size * i:batch_size * (i + 1)])
                query_xs = list(qx_t[batch_size * i:batch_size * (i + 1)])
                query_ys = list(qy_t[batch_size * i:batch_size * (i + 1)])
                # users_id也是按顺序的

                u_ids = users_id[batch_size * i:batch_size * (i + 1)]
            except IndexError:
                print("**********batch dataset index error!***********")
                continue

            # global update of one batch dataset
            # if u_ids:
            if dynamic_center:
                dynamic_p = cluster.get_dynamic_prefer(get_current_profile(users_profile, u_ids),
                                                         dynamic_center)  # (self, c_user_profile, task_dynamic_data):
            else:
                dynamic_p = []
            loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'], dynamic_p, u_ids)
            # else:
            #     loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

            b_loss.append(loss)
            batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys,  type="query", u_ids=u_ids,
                                               dynamic_p=dynamic_p)
            ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
            # if model_save:
            #     torch.save(model.state_dict(), model_filename)  # save save_model parameter
            #     # *************************************************************
            #     if reg_name:
            #         reg = {"omega": model.omega, "last_grad": model.last_grad}
            #         torch.save(reg, reg_name)
            #     if rep_name:
            #         # rep = {'u_reps': model.representation}
            #         # torch.save(rep, rep_name)
            #         torch.save(model.representation, rep_name)

            # test on history data set

            # # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration
            # batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys, type="query", u_ids=users_id)
            # ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
            #
            b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
            b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
            b_mae.append(sum(batch_mae) / len(batch_mae))
        if model_save:
            torch.save(model.state_dict(), model_filename)  # save save_model parameter
            # *************************************************************
            if reg_name:
                reg = {"omega": model.omega, "last_grad": model.last_grad}
                torch.save(reg, reg_name)
            if rep_name:
                # rep = {'u_reps': model.representation}
                # torch.save(rep, rep_name)
                torch.save(model.representation, rep_name)
        # 一个epoch测试一次
        try:
            #torch.load(model_filename)
            trained_state_dict = torch.load(model_filename)
            model.load_state_dict(trained_state_dict)
            reg = torch.load(reg_name)
            model.omega = reg['omega']
            model.last_grad = reg["last_grad"]
            rep = torch.load(rep_name)
            model.representation = rep
        except:
            raise "no training model to eval!"
        # batch_ndcg, batch_mae = evaluation(model, sx_t, sy_t, qx_t, qy_t, type="query", u_ids=users_id,dynamic_p=dynamic_p)
        # ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
        # b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
        # b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
        # b_mae.append(sum(batch_mae) / len(batch_mae))

        # test on cold users
        if dynamic_center:
            c_dynamic_p = cluster.get_dynamic_prefer(get_current_profile(users_profile, cus_id), dynamic_center)      # (self, c_user_profile, task_dynamic_data):
        else:
            c_dynamic_p = []
        cold_batch_ndcg, cold_batch_mae = evaluation(model, list(tr_x), list(tr_y), list(te_x), list(te_y), u_ids=cus_id, dynamic_p=c_dynamic_p)
        cndcg_1, cndcg_3 = cold_batch_ndcg[0], cold_batch_ndcg[1]

        c_epoch_ndcg_1.append(sum(cndcg_1) / len(cndcg_1))
        c_epoch_ndcg_3.append(sum(cndcg_3) / len(cndcg_3))
        c_epoch_mae.append(sum(cold_batch_mae) / len(cold_batch_mae))

        # TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
        #                          {"cndcg1": sum(cndcg_1) / len(te_y), 'cndcg3': sum(cndcg_3) / len(te_y)}, en)
        # TensorWriter.add_scalars('mae_{}'.format(stage_id),
        #                          {'mae': sum(cold_batch_mae) / len(te_y)}, en)

        epoch_ndcg_1.append(sum(b_ndcg_1) / len(b_ndcg_1))
        epoch_ndcg_3.append(sum(b_ndcg_3) / len(b_ndcg_3))
        epoch_mae.append(sum(b_mae) / len(b_mae))

        # 记录每个stage中各个epoch训练的收敛情况
        TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
                                 {"ndcg1": sum(b_ndcg_1) / len(b_ndcg_1), 'ndcg3': sum(b_ndcg_3) / len(b_ndcg_3),"cndcg1": sum(cndcg_1) / len(cndcg_1), 'cndcg3': sum(cndcg_3) / len(cndcg_3)}, en)

        TensorWriter.add_scalars('mae_{}'.format(stage_id),
                                 {'mae': sum(b_mae) / len(b_mae),'cmae':sum(cold_batch_mae) / len(cold_batch_mae)}, en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)

    # get/update user dynamic preference
    all_inters = [torch.cat([sx_t[i][1], qx_t[i][1]], 0) for i in range(len(sx_t))]
    all_y = [torch.cat([sy_t[i], qy_t[i]], 0) for i in range(len(sy_t))]
    print("user num and task num",len(users_id),len(all_inters))
    if dynamic_data:
        for i in range(len(all_inters)):
            item_emb = model.basemodel.item_embedding(model.item_load(all_inters[i]))
            dynamic_prefer = get_agg_item(item_emb, all_y[i])
            if dynamic_data.get(users_id[i])!=None:
                dynamic_data[users_id[i]] = 0.5 * dynamic_prefer + 0.5 * dynamic_data[users_id[i]]  #  或许可以设计attention
            else:
                dynamic_data[users_id[i]] = dynamic_prefer
        dynamic_center = cluster.get_dynamic_center(dynamic_data)
    else:
        dynamic_data = OrderedDict()
        for i in range(len(all_inters)):
            item_emb = model.basemodel.item_embedding(model.item_load(all_inters[i]))
            dynamic_prefer = get_agg_item(item_emb, all_y[i])
            dynamic_data[users_id[i]] = dynamic_prefer
        dynamic_center = cluster.get_dynamic_center(dynamic_data)
    torch.save({"dynamic_data":dynamic_data, "dynamic_center":dynamic_center}, "./save_model/dynamic_center.pkl")




        # TensorWriter.add_scalars('NDCG@10_HR@10_{}'.format(stage_id),{"ndcg": sum(b_ndcg_10) / batch_num, 'hr': sum(b_hr_10) / batch_num}, en)
    print("**********average NDCG@1on query set*********", sum(epoch_ndcg_1) / epoch_num)
    print("**********NDCG@1 on query set*********", epoch_ndcg_1)
    print("**********average NDCG@3 on query set*********", sum(epoch_ndcg_3) / epoch_num)
    print("**********NDCG@3 on query set*********", epoch_ndcg_3)
    print("**********average MAE on query set*********", sum(epoch_mae) / epoch_num)
    print("*********each epoch NDCG@1 on cold users***********", c_epoch_ndcg_1)
    print("**********average NDCG@1 on cold users*********", sum(c_epoch_ndcg_1) / epoch_num)
    print("*********each epoch NDCG@3 on cold users***********", c_epoch_ndcg_3)
    print("**********average NDCG@3 on cold users*********", sum(c_epoch_ndcg_3) / epoch_num)
    print("**********average MAE on cold users*********", sum(c_epoch_mae) / epoch_num)

    print("*********each epoch performance on cold users***********", c_epoch_ndcg_1[-1],c_epoch_ndcg_3[-1],c_epoch_mae[-1])

    if model_save:
        torch.save(model.state_dict(), model_filename)  # save save_model parameter
        # *************************************************************
        if reg_name:
            reg = {"omega": model.omega, "last_grad": model.last_grad}
            torch.save(reg, reg_name)
        if rep_name:
            #rep = {'u_reps': model.representation}
            #torch.save(rep, rep_name)
            torch.save(model.representation, rep_name)
    return epoch_ndcg_1[-1], c_epoch_ndcg_1[-1], epoch_ndcg_3[-1], c_epoch_ndcg_3[-1], \
           epoch_mae[-1], c_epoch_mae[-1]

    # return sum(epoch_ndcg_1) / epoch_num, sum(c_epoch_ndcg_1) / epoch_num, sum(epoch_ndcg_3) / epoch_num, sum(
    #     c_epoch_ndcg_3) / epoch_num, \
    #        sum(epoch_mae) / epoch_num, sum(c_epoch_mae) / epoch_num


def training0(model, current_dataset, test_dataset, epoch_num, batch_size, stage_id, model_save=True,
              model_filename=None, reg_name=None, users_id=None, rep_name=None):
    if config['use_cuda']:
        model.cuda()

    TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_log')
    # TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_full_log')
    # TensorWriter = SummaryWriter('datasets/movielens/baseline_performance/melu_log')

    training_set_size = len(current_dataset)
    print("*****************training_set_size", training_set_size)
    # batch_size = int(training_set_size / batch_num)
    batch_num = round(training_set_size / batch_size)
    print("*****************batch_num", batch_num)
    print("*****************batch_size", batch_size)
    model.train()

    tr_x, tr_y, te_x, te_y = zip(*test_dataset)

    epoch_ndcg_1, epoch_ndcg_3, c_epoch_ndcg_1, c_epoch_ndcg_3 = [], [], [], []
    epoch_mae, c_epoch_mae = [], []
    # epoch_hr_5,c_epoch_hr_5,epoch_hr_10,c_epoch_hr_10=[],[],[],[]
    for en in range(epoch_num):
        # random.shuffle(current_dataset)#打乱列表
        a, b, c, d = zip(*current_dataset)
        b_ndcg_1, cb_ndcg_1, b_ndcg_3, cb_ndcg_3 = [], [], [], []
        b_mae, cb_mae = [], []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])  # 取batch_size个用户训练
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
                # users_id也是按顺序的

                u_ids = users_id[batch_size * i:batch_size * (i + 1)]
            except IndexError:
                print("**********batch dataset index error!***********")
                continue

            # global update of one batch dataset
            # if u_ids:
            loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'], u_ids)
            # else:
            #     loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

            b_loss.append(loss)

            # test on history data set

            # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration

            batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys, type="query")
            ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
            b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
            b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
            b_mae.append(sum(batch_mae) / len(batch_mae))

            # test on cold users
            cold_batch_ndcg, cold_batch_mae = evaluation(model, list(tr_x), list(tr_y), list(te_x), list(te_y))
            cndcg_1, cndcg_3 = cold_batch_ndcg[0], cold_batch_ndcg[1]
            cb_ndcg_1.append(sum(cndcg_1) / len(cndcg_1))
            cb_ndcg_3.append(sum(cndcg_3) / len(cndcg_3))
            cb_mae.append(sum(cold_batch_mae) / len(cold_batch_mae))

        epoch_ndcg_1.append(sum(b_ndcg_1) / len(b_ndcg_1))
        c_epoch_ndcg_1.append(sum(cb_ndcg_1) / len(cb_ndcg_1))
        epoch_ndcg_3.append(sum(b_ndcg_3) / len(b_ndcg_3))
        c_epoch_ndcg_3.append(sum(cb_ndcg_3) / len(cb_ndcg_3))
        epoch_mae.append(sum(b_mae) / len(b_mae))
        c_epoch_mae.append(sum(cb_mae) / len(cb_mae))

        # 记录每个stage中各个epoch训练的收敛情况
        TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
                                 {"ndcg1": sum(b_ndcg_1) / len(b_ndcg_1), 'ndcg3': sum(b_ndcg_3) / len(b_ndcg_3),
                                  "cndcg1": sum(cb_ndcg_1) / len(cb_ndcg_1), 'cndcg3': sum(cb_ndcg_3) / len(cb_ndcg_3)},
                                 en)

        TensorWriter.add_scalars('mae_{}'.format(stage_id),
                                 {'mae': sum(b_mae) / len(b_mae), 'cmae': sum(cb_mae) / len(cb_mae)},
                                 en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)

        # TensorWriter.add_scalars('NDCG@10_HR@10_{}'.format(stage_id),{"ndcg": sum(b_ndcg_10) / batch_num, 'hr': sum(b_hr_10) / batch_num}, en)
    print("**********average NDCG@1on query set*********", sum(epoch_ndcg_1) / epoch_num)
    print("**********NDCG@1 on query set*********", epoch_ndcg_1)
    print("**********average NDCG@3 on query set*********", sum(epoch_ndcg_3) / epoch_num)
    print("**********NDCG@3 on query set*********", epoch_ndcg_3)
    print("**********average MAE on query set*********", sum(epoch_mae) / epoch_num)
    print("*********each epoch NDCG@1 on cold users***********", c_epoch_ndcg_1)
    print("**********average NDCG@1 on cold users*********", sum(c_epoch_ndcg_1) / epoch_num)
    print("*********each epoch NDCG@3 on cold users***********", c_epoch_ndcg_3)
    print("**********average NDCG@3 on cold users*********", sum(c_epoch_ndcg_3) / epoch_num)
    print("**********MAE on cold users*********", c_epoch_mae)
    print("**********average MAE on cold users*********", sum(c_epoch_mae) / epoch_num)

    if model_save:
        torch.save(model.state_dict(), model_filename)  # save save_model parameter
        # *************************************************************
        if reg_name:
            reg = {"omega": model.omega, "last_grad": model.last_grad}
            torch.save(reg, reg_name)
    return epoch_ndcg_1[-1], c_epoch_ndcg_1[-1], epoch_ndcg_3[-1], c_epoch_ndcg_3[-1], epoch_mae[-1], c_epoch_mae[-1]
def training_amazon(model, current_dataset, test_dataset, epoch_num, batch_size, stage_id, model_save=True,
             model_filename=None, reg_name=None, users_id=None, cus_id=None, rep_name=None, dynamic_center= None, dynamic_data=None):
    TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_log')
    if config['use_cuda']:
        model.cuda()

    '''get current user cluster based on user's profile'''
    # cluster(get_current_profile(users_profile, users_id), config['cluster_n'])

    training_set_size = len(current_dataset)
    print("*****************training_set_size", training_set_size)
    # batch_size = int(training_set_size / batch_num)
    batch_num = round(training_set_size / batch_size)
    print("*****************batch_num", batch_num)
    print("*****************batch_size", batch_size)
    model.train()

    tr_x, tr_y, te_x, te_y = zip(*test_dataset)
    sx_t, sy_t, qx_t, qy_t = zip(*current_dataset)


    epoch_ndcg_1, epoch_ndcg_3, c_epoch_ndcg_1, c_epoch_ndcg_3 = [], [], [], []
    epoch_mae, c_epoch_mae = [], []
    # epoch_hr_5,c_epoch_hr_5,epoch_hr_10,c_epoch_hr_10=[],[],[],[]
    for en in range(epoch_num):
        # random.shuffle(current_dataset)#打乱列表
        b_ndcg_1,  b_ndcg_3,  = [], []
        b_mae = []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(sx_t[batch_size * i:batch_size * (i + 1)])       # 取batch_size个用户训练
                supp_ys = list(sy_t[batch_size * i:batch_size * (i + 1)])
                query_xs = list(qx_t[batch_size * i:batch_size * (i + 1)])
                query_ys = list(qy_t[batch_size * i:batch_size * (i + 1)])
                # users_id也是按顺序的

                u_ids = users_id[batch_size * i:batch_size * (i + 1)]
            except IndexError:
                print("**********batch dataset index error!***********")
                continue

            # global update of one batch dataset
            # if u_ids:

            loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'], u_ids)
            # else:
            #     loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

            b_loss.append(loss)
            batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys,  type="query", u_ids=u_ids)
            if not batch_ndcg: continue

            ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
            # if model_save:
            #     torch.save(model.state_dict(), model_filename)  # save save_model parameter
            #     # *************************************************************
            #     if reg_name:
            #         reg = {"omega": model.omega, "last_grad": model.last_grad}
            #         torch.save(reg, reg_name)
            #     if rep_name:
            #         # rep = {'u_reps': model.representation}
            #         # torch.save(rep, rep_name)
            #         torch.save(model.representation, rep_name)

            # test on history data set

            # # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration
            # batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys, type="query", u_ids=users_id)
            # ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
            #
            b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
            b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
            b_mae.append(sum(batch_mae) / len(batch_mae))
        if model_save:
            torch.save(model.state_dict(), model_filename)  # save save_model parameter
            # *************************************************************
            if reg_name:
                reg = {"omega": model.omega, "last_grad": model.last_grad}
                torch.save(reg, reg_name)
            if rep_name:
                # rep = {'u_reps': model.representation}
                # torch.save(rep, rep_name)
                torch.save(model.representation, rep_name)
        # 一个epoch测试一次
        try:
            #torch.load(model_filename)
            trained_state_dict = torch.load(model_filename)
            model.load_state_dict(trained_state_dict)
            reg = torch.load(reg_name)
            model.omega = reg['omega']
            model.last_grad = reg["last_grad"]
            rep = torch.load(rep_name)
            model.representation = rep
        except:
            raise "no training model to eval!"
        # batch_ndcg, batch_mae = evaluation(model, sx_t, sy_t, qx_t, qy_t, type="query", u_ids=users_id,dynamic_p=dynamic_p)
        # ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
        # b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
        # b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
        # b_mae.append(sum(batch_mae) / len(batch_mae))

        # test on cold users

        cold_batch_ndcg, cold_batch_mae = evaluation(model, list(tr_x), list(tr_y), list(te_x), list(te_y), u_ids=cus_id)
        cndcg_1, cndcg_3 = cold_batch_ndcg[0], cold_batch_ndcg[1]

        c_epoch_ndcg_1.append(sum(cndcg_1) / len(cndcg_1))
        c_epoch_ndcg_3.append(sum(cndcg_3) / len(cndcg_3))
        c_epoch_mae.append(sum(cold_batch_mae) / len(cold_batch_mae))

        # TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
        #                          {"cndcg1": sum(cndcg_1) / len(te_y), 'cndcg3': sum(cndcg_3) / len(te_y)}, en)
        # TensorWriter.add_scalars('mae_{}'.format(stage_id),
        #                          {'mae': sum(cold_batch_mae) / len(te_y)}, en)

        epoch_ndcg_1.append(sum(b_ndcg_1) / len(b_ndcg_1))
        epoch_ndcg_3.append(sum(b_ndcg_3) / len(b_ndcg_3))
        epoch_mae.append(sum(b_mae) / len(b_mae))

        # 记录每个stage中各个epoch训练的收敛情况
        TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
                                 {"ndcg1": sum(b_ndcg_1) / len(b_ndcg_1), 'ndcg3': sum(b_ndcg_3) / len(b_ndcg_3),"cndcg1": sum(cndcg_1) / len(cndcg_1), 'cndcg3': sum(cndcg_3) / len(cndcg_3)}, en)

        TensorWriter.add_scalars('mae_{}'.format(stage_id),
                                 {'mae': sum(b_mae) / len(b_mae),'cmae':sum(cold_batch_mae) / len(cold_batch_mae)}, en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)

    # get/update user dynamic preference
    all_inters = [torch.cat([sx_t[i][1], qx_t[i][1]], 0) for i in range(len(sx_t))]
    all_y = [torch.cat([sy_t[i], qy_t[i]], 0) for i in range(len(sy_t))]
    print("user num and task num",len(users_id),len(all_inters))


        # TensorWriter.add_scalars('NDCG@10_HR@10_{}'.format(stage_id),{"ndcg": sum(b_ndcg_10) / batch_num, 'hr': sum(b_hr_10) / batch_num}, en)
    print("**********average NDCG@1on query set*********", sum(epoch_ndcg_1) / epoch_num)
    print("**********NDCG@1 on query set*********", epoch_ndcg_1)
    print("**********average NDCG@3 on query set*********", sum(epoch_ndcg_3) / epoch_num)
    print("**********NDCG@3 on query set*********", epoch_ndcg_3)
    print("**********average MAE on query set*********", sum(epoch_mae) / epoch_num)
    print("*********each epoch NDCG@1 on cold users***********", c_epoch_ndcg_1)
    print("**********average NDCG@1 on cold users*********", sum(c_epoch_ndcg_1) / epoch_num)
    print("*********each epoch NDCG@3 on cold users***********", c_epoch_ndcg_3)
    print("**********average NDCG@3 on cold users*********", sum(c_epoch_ndcg_3) / epoch_num)
    print("**********average MAE on cold users*********", sum(c_epoch_mae) / epoch_num)

    print("*********each epoch performance on cold users***********", c_epoch_ndcg_1[-1],c_epoch_ndcg_3[-1],c_epoch_mae[-1])

    if model_save:
        torch.save(model.state_dict(), model_filename)  # save save_model parameter
        # *************************************************************
        if reg_name:
            reg = {"omega": model.omega, "last_grad": model.last_grad}
            torch.save(reg, reg_name)
        if rep_name:
            #rep = {'u_reps': model.representation}
            #torch.save(rep, rep_name)
            torch.save(model.representation, rep_name)
    return epoch_ndcg_1[-1], c_epoch_ndcg_1[-1], epoch_ndcg_3[-1], c_epoch_ndcg_3[-1], \
           epoch_mae[-1], c_epoch_mae[-1]

    # return sum(epoch_ndcg_1) / epoch_num, sum(c_epoch_ndcg_1) / epoch_num, sum(epoch_ndcg_3) / epoch_num, sum(
    #     c_epoch_ndcg_3) / epoch_num, \
    #        sum(epoch_mae) / epoch_num, sum(c_epoch_mae) / epoch_num

def get_agg_item(items, scores):
    weights = torch.nn.functional.softmax(scores, dim=0)
    i_att = [items[i] * weights[i].item() for i in range(len(scores))]
    i_att = torch.mean(torch.stack(i_att), dim=0)
    return i_att  # 作为该用户的dynamic_preferences
def meta_update(model, current_dataset, test_dataset, epoch_num, batch_size, stage_id, model_save=True,
             model_filename=None, reg_name=None, users_id=None, cus_id=None, rep_name=None):
    if config['use_cuda']:
        model.cuda()

    TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_log')
    # TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_full_log')
    #TensorWriter = SummaryWriter('datasets/movielens/baseline_performance/melu_log')

    training_set_size = len(current_dataset)
    print("*****************training_set_size", training_set_size)
    # batch_size = int(training_set_size / batch_num)
    batch_num = round(training_set_size / batch_size)
    print("*****************batch_num", batch_num)
    print("*****************batch_size", batch_size)
    model.train()

    tr_x, tr_y, te_x, te_y = zip(*test_dataset)

    epoch_ndcg_1, epoch_ndcg_3, c_epoch_ndcg_1, c_epoch_ndcg_3 = [], [], [], []
    epoch_mae, c_epoch_mae = [], []
    # epoch_hr_5,c_epoch_hr_5,epoch_hr_10,c_epoch_hr_10=[],[],[],[]
    for en in range(epoch_num):
        # random.shuffle(current_dataset)#打乱列表
        a, b, c, d = zip(*current_dataset)
        b_ndcg_1,  b_ndcg_3,  = [], []
        b_mae = []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])       # 取batch_size个用户训练
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
                # users_id也是按顺序的

                u_ids = users_id[batch_size * i:batch_size * (i + 1)]
            except IndexError:
                print("**********batch dataset index error!***********")
                continue

            # global update of one batch dataset
            # if u_ids:
            loss = model.meta_test(supp_xs, supp_ys, query_xs, query_ys, u_ids)
            # else:
            #     loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])

            b_loss.append(loss)

            # test on history data set

            # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration

            batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys, type="query", u_ids=users_id)
            ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]

            b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
            b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
            b_mae.append(sum(batch_mae) / len(batch_mae))

        # test on cold users
        cold_batch_ndcg, cold_batch_mae = evaluation(model, list(tr_x), list(tr_y), list(te_x), list(te_y), u_ids=cus_id)
        cndcg_1, cndcg_3 = cold_batch_ndcg[0], cold_batch_ndcg[1]

        c_epoch_ndcg_1.append(sum(cndcg_1) / len(cndcg_1))
        c_epoch_ndcg_3.append(sum(cndcg_3) / len(cndcg_3))
        c_epoch_mae.append(sum(cold_batch_mae) / len(cold_batch_mae))

        # TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
        #                          {"cndcg1": sum(cndcg_1) / len(te_y), 'cndcg3': sum(cndcg_3) / len(te_y)}, en)
        # TensorWriter.add_scalars('mae_{}'.format(stage_id),
        #                          {'mae': sum(cold_batch_mae) / len(te_y)}, en)

        epoch_ndcg_1.append(sum(b_ndcg_1) / len(b_ndcg_1))
        epoch_ndcg_3.append(sum(b_ndcg_3) / len(b_ndcg_3))
        epoch_mae.append(sum(b_mae) / len(b_mae))

        # 记录每个stage中各个epoch训练的收敛情况
        TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
                                 {"ndcg1": sum(b_ndcg_1) / len(b_ndcg_1), 'ndcg3': sum(b_ndcg_3) / len(b_ndcg_3),"cndcg1": sum(cndcg_1) / len(cndcg_1), 'cndcg3': sum(cndcg_3) / len(cndcg_3)}, en)

        TensorWriter.add_scalars('mae_{}'.format(stage_id),
                                 {'mae': sum(b_mae) / len(b_mae),'cmae':sum(cold_batch_mae) / len(cold_batch_mae)}, en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)


        # TensorWriter.add_scalars('NDCG@10_HR@10_{}'.format(stage_id),{"ndcg": sum(b_ndcg_10) / batch_num, 'hr': sum(b_hr_10) / batch_num}, en)
    print("**********average NDCG@1on query set*********", sum(epoch_ndcg_1) / epoch_num)
    print("**********NDCG@1 on query set*********", epoch_ndcg_1)
    print("**********average NDCG@3 on query set*********", sum(epoch_ndcg_3) / epoch_num)
    print("**********NDCG@3 on query set*********", epoch_ndcg_3)
    print("**********average MAE on query set*********", sum(epoch_mae) / epoch_num)
    print("*********each epoch NDCG@1 on cold users***********", c_epoch_ndcg_1)
    print("**********average NDCG@1 on cold users*********", sum(c_epoch_ndcg_1) / epoch_num)
    print("*********each epoch NDCG@3 on cold users***********", c_epoch_ndcg_3)
    print("**********average NDCG@3 on cold users*********", sum(c_epoch_ndcg_3) / epoch_num)
    print("**********average MAE on cold users*********", sum(c_epoch_mae) / epoch_num)

    if model_save:
        torch.save(model.state_dict(), model_filename)  # save save_model parameter
        # *************************************************************
        if reg_name:
            reg = {"omega": model.omega, "last_grad": model.last_grad}
            torch.save(reg, reg_name)
        if rep_name:
            #rep = {'u_reps': model.representation}
            #torch.save(rep, rep_name)
            torch.save(model.representation, rep_name)

    return sum(epoch_ndcg_1) / epoch_num, sum(c_epoch_ndcg_1) / epoch_num, sum(epoch_ndcg_3) / epoch_num, sum(
        c_epoch_ndcg_3) / epoch_num, \
           sum(epoch_mae) / epoch_num, sum(c_epoch_mae) / epoch_num


def training2(model, current_dataset, test_dataset, epoch_num, batch_size, stage_id, model_save=True,
             model_filename=None):
    #if config['use_cuda']:
    #    model.cuda()

    TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/melu')
    # TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_full_log')
    # TensorWriter = SummaryWriter('datasets/movielens/baseline_performance/form')

    training_set_size = len(current_dataset)
    print("*****************training_set_size", training_set_size)
    # batch_size = int(training_set_size / batch_num)
    batch_num = max(round(training_set_size / batch_size),1)
    print("*****************batch_num", batch_num)
    print("*****************batch_size", batch_size)
    model.train()

    tr_x, tr_y, te_x, te_y = zip(*test_dataset)

    epoch_ndcg_1, epoch_ndcg_3, c_epoch_ndcg_1, c_epoch_ndcg_3 = [], [], [], []
    epoch_mae, c_epoch_mae = [], []
    # epoch_hr_5,c_epoch_hr_5,epoch_hr_10,c_epoch_hr_10=[],[],[],[]
    loss_change = 10  # 收敛条件为loss_change<0.005
    last_loss=20
    flag = False
    for en in range(epoch_num):
        # random.shuffle(current_dataset)#打乱列表
        a, b, c, d = zip(*current_dataset)
        b_ndcg_1, cb_ndcg_1, b_ndcg_3, cb_ndcg_3 = [], [], [], []
        b_mae, cb_mae = [], []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])       # 取batch_size个用户训练
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
            except IndexError:
                print("**********batch dataset index error!***********")
                continue

            # global update of one batch dataset
            loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])
            b_loss.append(loss)

            # test on history data set

            # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration
            batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys, type="query")
            ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]
            b_ndcg_1.append(sum(ndcg_1) / len(ndcg_1))
            b_ndcg_3.append(sum(ndcg_3) / len(ndcg_3))
            b_mae.append(sum(batch_mae) / len(batch_mae))

            # test on cold users
        print(b_loss)


        cold_batch_ndcg, cold_batch_mae = evaluation(model, list(tr_x), list(tr_y), list(te_x), list(te_y))
        cndcg_1, cndcg_3 = cold_batch_ndcg[0], cold_batch_ndcg[1]
        # loss_change = abs(last_loss - sum(cold_batch_mae) / len(cold_batch_mae))
        # last_loss = sum(cold_batch_mae) / len(cold_batch_mae)

        c_epoch_ndcg_1.append(sum(cndcg_1) / len(cndcg_1))
        c_epoch_ndcg_3.append(sum(cndcg_3) / len(cndcg_3))
        c_epoch_mae.append(sum(cold_batch_mae) / len(cold_batch_mae))

        # TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
        #                          {"cndcg1": sum(cndcg_1) / len(te_y), 'cndcg3': sum(cndcg_3) / len(te_y)}, en)
        # TensorWriter.add_scalars('mae_{}'.format(stage_id),
        #                          {'mae': sum(cold_batch_mae) / len(te_y)}, en)

        epoch_ndcg_1.append(sum(b_ndcg_1) / len(b_ndcg_1))
        epoch_ndcg_3.append(sum(b_ndcg_3) / len(b_ndcg_3))
        epoch_mae.append(sum(b_mae) / len(b_mae))

        # 记录每个stage中各个epoch训练的收敛情况
        # TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        # TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
        #                          {"ndcg1": sum(b_ndcg_1) / len(b_ndcg_1), 'ndcg3': sum(b_ndcg_3) / len(b_ndcg_3),
        #                           "cndcg1": sum(cndcg_1) / len(cndcg_1), 'cndcg3': sum(cndcg_3) / len(cndcg_3)}, en)
        #
        # TensorWriter.add_scalars('mae_{}'.format(stage_id),
        #                          {'mae': sum(b_mae) / len(b_mae), 'cmae': sum(cold_batch_mae) / len(cold_batch_mae)},
        #                          en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)

    print("**********average NDCG@1on query set*********", sum(epoch_ndcg_1) / epoch_num)
    print("**********NDCG@1 on query set*********", epoch_ndcg_1)
    print("**********average NDCG@3 on query set*********", sum(epoch_ndcg_3) / epoch_num)
    print("**********NDCG@3 on query set*********", epoch_ndcg_3)
    print("**********average MAE on query set*********", sum(epoch_mae) / epoch_num)
    print("*********each epoch NDCG@1 on cold users***********", c_epoch_ndcg_1)
    print("**********average NDCG@1 on cold users*********", sum(c_epoch_ndcg_1) / epoch_num)
    print("*********each epoch NDCG@3 on cold users***********", c_epoch_ndcg_3)
    print("**********average NDCG@3 on cold users*********", sum(c_epoch_ndcg_3) / epoch_num)
    print("********** MAE on cold users*********", c_epoch_mae)
    print("**********average MAE on cold users*********", sum(c_epoch_mae) / epoch_num)
    if 0<=abs(c_epoch_mae[-2]-c_epoch_mae[-1])<=0.003 and 0<=abs(c_epoch_mae[-3]-c_epoch_mae[-2])<=0.003: #超参数，看设置多少合适
        flag=True
    if model_save:
        torch.save(model.state_dict(), model_filename)  # save save_model parameter
    return sum(epoch_ndcg_1) / epoch_num, sum(c_epoch_ndcg_1) / epoch_num, sum(epoch_ndcg_3) / epoch_num, sum(
        c_epoch_ndcg_3) / epoch_num, \
           sum(epoch_mae) / epoch_num, sum(c_epoch_mae) / epoch_num, flag

def training3(model, current_d, next_d , test_d, args, epoch_num, batch_size, stage_id, model_save=True, model_filename=None):
    if config['use_cuda']:
        model.cuda()
    TensorWriter = SummaryWriter('datasets/movielens/baseline_performance/sml_log')

    training_set_size = len(current_d)
    print("*****************training_set_size", training_set_size)
    # batch_size = int(training_set_size / batch_num)
    batch_num = round(training_set_size / batch_size)
    print("*****************batch_num", batch_num)
    print("*****************batch_size", batch_size)


    tr_x, tr_y, te_x, te_y = zip(*test_d)

    epoch_ndcg_1, epoch_ndcg_3, c_epoch_ndcg_1, c_epoch_ndcg_3 = [], [], [], []
    epoch_mae, c_epoch_mae = [], []
    # epoch_hr_5,c_epoch_hr_5,epoch_hr_10,c_epoch_hr_10=[],[],[],[]
    cur_sx, cur_sy, cur_nx, cur_qx, cur_qy = zip(*current_d)
    next_sx,next_sy,next_nx,next_qx,next_qy = zip(*current_d)
    for en in range(args.MF_epochs):
        model.MFbase.train()
        model.transfer.eval()
        # random.shuffle(current_dataset)#打乱列表
        b_ndcg_1, cb_ndcg_1, b_ndcg_3, cb_ndcg_3 = [], [], [], []
        b_mae, cb_mae = [], []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(cur_sx[batch_size * i:batch_size * (i + 1)])       # 取batch_size个用户训练
                supp_ys = list(cur_sy[batch_size * i:batch_size * (i + 1)])
                query_xs = list(cur_qx[batch_size * i:batch_size * (i + 1)])
                query_ys = list(cur_qy[batch_size * i:batch_size * (i + 1)])
                neg_xs = list(cur_nx[batch_size * i:batch_size * (i + 1)])
            except IndexError:
                print("**********batch dataset index error!***********")
                continue
            # global update of one batch dataset
            #loss = model.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])
            # set_t = [supp_xs,supp_ys,neg_xs,query_xs,query_ys]
            # set_tt =


            # loss = model.train_one_stage3(args, stage_id, set_t, set_tt, [query_xs,query_ys])#train_one_stage3(self,args,stage_id, set_t, set_tt, now_test,val):
            #loss = model.MF_train_onestage(args, supp_xs, supp_ys, neg_xs, query_xs,query_ys)    # args, set_t, stage_id, val=None

            # b_loss.append(loss)

            # test on history data set

            # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration
            if len(query_ys) > 3:
                batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys, type="query")
                ndcg_1, ndcg_3 = batch_ndcg[0], batch_ndcg[1]

                b_ndcg_1.append(sum(ndcg_1) / len(query_ys))
                b_ndcg_3.append(sum(ndcg_3) / len(query_ys))
                b_mae.append(sum(batch_mae) / len(query_ys))

            # test on cold users
            if len(list(te_y)) > 3:
                cold_batch_ndcg, cold_batch_mae = evaluation(model, list(tr_x), list(tr_y), list(te_x), list(te_y))
                cndcg_1, cndcg_3 = cold_batch_ndcg[0], cold_batch_ndcg[1]

                cb_ndcg_1.append(sum(cndcg_1) / len(te_y))
                cb_ndcg_3.append(sum(cndcg_3) / len(te_y))
                cb_mae.append(sum(cold_batch_mae) / len(te_y))

        epoch_ndcg_1.append(sum(b_ndcg_1) / len(b_ndcg_1))
        c_epoch_ndcg_1.append(sum(cb_ndcg_1) / len(cb_ndcg_1))
        epoch_ndcg_3.append(sum(b_ndcg_3) / len(b_ndcg_3))
        c_epoch_ndcg_3.append(sum(cb_ndcg_3) / len(cb_ndcg_3))
        epoch_mae.append(sum(b_mae) / len(b_mae))
        c_epoch_mae.append(sum(cb_mae) / len(cb_mae))

        #记录每个stage中各个epoch训练的收敛情况
        TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        TensorWriter.add_scalars('NDCG@1_NDCG@3_{}'.format(stage_id),
                                 {"ndcg1": sum(cb_ndcg_1) / batch_num, 'ndcg3': sum(cb_ndcg_3) / batch_num}, en)
        # TensorWriter.add_scalars('NDCG@10_HR@10_{}'.format(stage_id),{"ndcg": sum(b_ndcg_10) / batch_num, 'hr': sum(b_hr_10) / batch_num}, en)
        TensorWriter.add_scalar('mae_{}'.format(stage_id), sum(cb_mae) / batch_num,
                                en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)
    print("**********average NDCG@1on query set*********", sum(epoch_ndcg_1) / epoch_num)
    print("**********NDCG@1 on query set*********", epoch_ndcg_1)

    print("**********average NDCG@3 on query set*********", sum(epoch_ndcg_3) / epoch_num)
    print("**********NDCG@3 on query set*********", epoch_ndcg_3)

    print("**********average MAE on query set*********", sum(epoch_mae) / epoch_num)

    print("*********each epoch NDCG@1 on cold users***********", c_epoch_ndcg_1)
    print("**********average NDCG@1 on cold users*********", sum(c_epoch_ndcg_1) / epoch_num)
    print("*********each epoch NDCG@3 on cold users***********", c_epoch_ndcg_3)
    print("**********average NDCG@3 on cold users*********", sum(c_epoch_ndcg_3) / epoch_num)

    print("**********average MAE on cold users*********", sum(c_epoch_mae) / epoch_num)

    if model_save:
        torch.save(model.state_dict(), model_filename)  # save save_model parameter
    return sum(epoch_ndcg_1) / epoch_num, sum(c_epoch_ndcg_1) / epoch_num, sum(epoch_ndcg_3) / epoch_num, sum(
        c_epoch_ndcg_3) / epoch_num, \
           sum(epoch_mae) / epoch_num, sum(c_epoch_mae) / epoch_num

def test(model, current_dataset, test_dataset, epoch_num, batch_num, stage_id):
    if config['use_cuda']:
        model.cuda()

    TensorWriter = SummaryWriter('./datasets/movielens/log_test')

    training_set_size = len(current_dataset)
    print("*****************training_set_size", training_set_size)
    batch_size = int(training_set_size / batch_num)
    print("*****************batch_size", batch_size)
    model.train()

    tr_x, tr_y, te_x, te_y = zip(*test_dataset)

    epoch_ndcg, c_epoch_ndcg = [], []
    epoch_hr, c_epoch_hr = [], []
    for en in range(epoch_num):
        # random.shuffle(current_dataset)#打乱列表
        a, b, c, d = zip(*current_dataset)
        b_ndcg, cb_ndcg = [], []
        b_hr, cb_hr = [], []
        b_loss = []
        print("***************begin {} epoch training************".format(en))
        for i in tqdm(range(batch_num)):
            try:
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])  # 取batch_size个用户训练
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
            except IndexError:
                continue

            # global update

            # test_dataset used to test on cold users and record the preference of meta-save_model based on ndcg and hit ration
            batch_ndcg, batch_mae = evaluation(model, supp_xs, supp_ys, query_xs, query_ys)
            ndcg_5, ndcg_10 = batch_ndcg[0], batch_ndcg[1]

            b_ndcg_5.append(sum(ndcg_5) / len(query_ys))
            b_hr_5.append(sum(hr_5) / len(query_ys))
            b_ndcg_10.append(sum(ndcg_10) / len(query_ys))
            b_hr_10.append(sum(hr_10) / len(query_ys))

            # test on cold users
            cold_batch_ndcg, cold_batch_hr = evaluation(model, tr_x, tr_y, te_x, te_y)
            cndcg_5, cndcg_10 = cold_batch_ndcg[0], cold_batch_ndcg[1]
            chr_5, chr_10 = cold_batch_hr[0], cold_batch_hr[1]
            cb_ndcg.append(sum(cold_batch_ndcg) / len(te_y))
            cb_hr.append(sum(cold_batch_hr) / len(te_y))
            cb_ndcg.append(sum(cold_batch_ndcg) / len(te_y))
            cb_hr.append(sum(cold_batch_hr) / len(te_y))

        epoch_ndcg.append(sum(b_ndcg) / batch_num)
        epoch_hr.append(sum(b_hr) / batch_num)
        c_epoch_ndcg.append(sum(cb_ndcg) / batch_num)
        c_epoch_hr.append(sum(cb_hr) / batch_num)
        # TensorWriter.add_scalar('loss_{}'.format(stage_id), sum(b_loss) / batch_num, en)
        TensorWriter.add_scalars('NDCG@5_HR@5_{}'.format(stage_id),
                                 {"ndcg": sum(b_ndcg) / batch_num, 'hr': sum(b_hr) / batch_num}, en)
        # TensorWriter.add_scalar('HR@5_{}'.format(stage_id), sum(b_hr)/batch_num, en)  # 纵轴，横轴nt("*********each epoch NDCG on query set***********",epoch_ndcg)
    print("**********average NDCG on query set*********", sum(epoch_ndcg) / epoch_num)
    print("*********each epoch HR on query set***********", epoch_hr)
    print("**********average HR on query set*********", sum(epoch_hr) / epoch_num)

    print("*********each epoch NDCG on cold users***********", c_epoch_ndcg)
    print("**********average NDCG on cold users*********", sum(c_epoch_ndcg) / epoch_num)
    print("*********each epoch HR on cold users***********", c_epoch_hr)
    print("**********average HR on cold users*********", sum(c_epoch_hr) / epoch_num)
    return sum(epoch_ndcg) / epoch_num, sum(epoch_hr) / epoch_num, sum(c_epoch_ndcg) / epoch_num, sum(
        c_epoch_hr) / epoch_num


# 测试模型性能
def evaluation(model, train_x, train_y, test_x, test_y, type="cold", u_ids=None, dynamic_p=None):
    model.eval()
    batch_sz = len(test_y)
    batch_ndcg = []
    batch_mae = []
    if type == "cold" and config['use_cuda']:
        for i in range(batch_sz):
            train_x[i] = [train_x[i][0].cuda(), train_x[i][1].cuda()]
            train_y[i] = train_y[i].cuda()
            test_x[i] = [test_x[i][0].cuda(), test_x[i][1].cuda()]
            test_y[i] = test_y[i].cuda()

            # amazon
            # train_x[i] = train_x[i].cuda()
            # train_y[i] = train_y[i].cuda()
            # test_x[i] = test_x[i].cuda()
            # test_y[i] = test_y[i].cuda()
    for i in range(batch_sz):  # 一个batch
        # if u_ids:
        #     _,y_pred = model.meta_test(train_x[i], train_y[i], test_x[i], config['inner'], u_ids[i])
        # else:
        #       _,y_pred = model.meta_test(train_x[i], train_y[i], test_x[i], config['inner'])
        if dynamic_p:
            y_pred = model.meta_test(train_x[i], train_y[i], test_x[i], 4, dynamic_p[u_ids[i]]) # config['inner']
        else:
            # y_pred = model.meta_test(train_x[i], train_y[i], test_x[i], config['inner'])
            _, y_pred = model.forward(train_x[i], train_y[i], test_x[i], config['inner'])
        # print("************模型输出：",query_y_pred)
        if len(test_y[i]) < 5: continue
        n_dcg = ndcg(test_y[i]-1, y_pred)
        a_mae = mae(test_y[i]-1, y_pred)
        batch_ndcg.append(n_dcg)
        batch_mae.append(a_mae.item())

    return list(zip(*batch_ndcg)), batch_mae