import argparse
import os
import math
import pickle

import torch
import numpy as np
from copy import deepcopy
import sys

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
# 'num_actor': 8030,
from options import config,default_info,activation_func,get_params,get_zeros_like_params,init_params,get_grad

from torch.utils.data import Dataset
from data_process.dataset import online_movielens_1m,online_yelp
from data_process.input_loading import *
from data_process.embedding import ItemEmbedding,UserEmbedding

from collections import OrderedDict
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
TensorWriter = SummaryWriter('datasets/movielens/mepl_performance/mep_log')
TensorWriter2 = SummaryWriter('./datasets/movielens/log_test')

from options import config,default_info,activation_func,get_zeros_like_params,binary_conversion

from model_training import training,test
from Personal_lr import SelfAttention, Attention2, Attention, MultiheadAttention, item_Attention


class Recommender_model(torch.nn.Module):
    def __init__(self, emb_dim,layer_num,act_func,dataset_type='movielens',classification=False):
        super(Recommender_model, self).__init__()
        self.embedding_dim =emb_dim# config['embedding_dim']#初始化embedding layer
        self.input_size=self.embedding_dim*2
        self.layer_num=layer_num#config['layer_num']
        last_size=self.input_size
        fc_layers=[]
        act_func=act_func#config['activate_func']
        for i in range(self.layer_num-1):
            out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, out_dim)
            fc_layers.append(linear_model)
            last_size = out_dim
            fc_layers.append(activation_func(act_func))

        self.fc = torch.nn.Sequential(*fc_layers)
        self.device = torch.device('cuda' if config['use_cuda'] else "cpu")

        #self.active_func = config['active_func']
        # if dataset_type == 'movielens':
        #     # self.item_emb = ml_item_embedding(self.layer_num, default_info[dataset_type]['i_in_dim'] * self.embedding_dim,
        #     #                             self.embedding_dim, activation=activation_func('relu')).to(self.device)
        #     # self.user_emb = ml_user_embedding(self.layer_num, default_info[dataset_type]['u_in_dim'] * self.embedding_dim,
        #     #                             self.embedding_dim, activation=activation_func('relu')).to(self.device)  # n_layer, in_dim, embedding_dim, activation='sigmoid'
        #     self.item_emb = ml_item_embedding(config)
        #     self.user_emb = ml_user_embedding(config)
        # elif dataset_type == 'yelp':
        #     self.item_emb = yelp_item_embedding(self.layer_num, default_info[dataset_type]['i_in_dim'] * self.embedding_dim,
        #                                 self.embedding_dim, activation=self.active_func).to(self.device)
        #     self.user_emb = yelp_user_embedding(self.layer_num, default_info[dataset_type]['u_in_dim'] * self.embedding_dim,
        #                                 self.embedding_dim, activation=self.active_func).to(self.device)  # n_layer, in_dim, embedding_dim, activation='sigmoid'
        # else:
        #     self.item_emb = amazon_item_embedding(self.layer_num, default_info[dataset_type]['i_in_dim'] * self.embedding_dim,
        #                                 self.embedding_dim, activation=self.active_func).to(self.device)
        #     self.user_emb = amazon_user_embedding(self.layer_num, default_info[dataset_type]['u_in_dim'] * self.embedding_dim,
        #                                 self.embedding_dim, activation=self.active_func).to(self.device)  # n_layer, in_dim, embedding_dim, activation='sigmoid'


        # 根据任务的输出结果，决定最后一层的结构,n_y是items数量
        if classification:  # 推荐top_k，n_y表示finals需要输出几个，一般一个用户对一个movie输出一个评分,输出对所有Y标签的概率，选择最大的一个作为模型的预测输出
            finals = [torch.nn.Linear(last_size, default_info[dataset_type]['n_y']), activation_func('relu')]

        else:  # y的取值只有1个
            finals = [torch.nn.Linear(last_size, 1)]
        self.final_layer = torch.nn.Sequential(*finals)



    def forward(self, eu, ei):
        x = torch.cat([ei, eu], 1)  # catenete user_embedding and item_embedding
        out = self.fc(x)
        out = self.final_layer(out)

        #return eu, ei, out  # 输出user对item的评分
        return out


    # '''
    # red_model:推荐基模型、
    # ？？？是要对batch中的每个数据计算结果之后再得到topk还是model能直接得到topk?????
    # '''
    # def test(self, rec_model, inputs_data, topK=20):
    #     user = inputs_data[:, 0]  # x[:,n]表⽰在全部数组（维）中取第n个数
    #     pos_negs_items = inputs_data[:, 1:]  # all item
    #     user_embedding = self.user_emb(user)
    #     #user_bias = self.user_bais(user)
    #     pos_negs_items_embedding = self.item_emb(pos_negs_items)  # batch * 999 * item_laten
    #     #pos_negs_bias = self.item_bais(pos_negs_items)
    #
    #     user_embedding_ = user_embedding.unsqueeze(1)  # add a din to user embedding
    #     scores=rec_model(user,pos_negs_items)
    #
    #     #$pos_negs_interaction = torch.mul(user_embedding_, pos_negs_items_embedding).sum(-1)  # 基于MF的实例化应用，预测交互分数
    #     # pos_negs_interaction = torch.chain_matmul(user_embedding,pos_negs_items_embedding)  # we can use this code to compute the interaction
    #     #pos_negs_scores = pos_negs_interaction
    #     # the we have compute all score for pos and neg interactions ,each row has scorces of one pos inter and neg_num(99)  neg inter
    #     '''torch.topk(input:tensor,k:int,dim:int,largest:bool)
    #     dim=0表示按照列求topn，dim=1表示按照行求topK，None情况下，dim=1.其中largest=True表示从大到小取元素
    #     每行都求topK,取数组的前k个元素进行排序。
    #     用来获取张量或者数组中最大或者最小的元素以及索引位置
    #     通常该函数返回2个值：第一个值为排序的数组，得到的values是原数组dim=1的四组从大到小的三个元素值；
    #                         第二个值为该数组中获取到的元素在原数组中的位置标号，得到的indices是获取到的元素值在原数组dim=1中的位置。
    #     '''
    #     #########要不要用ndcg(ground_truth, test_result, top_k=3):
    #     _, rank = torch.topk(scores, topK)  # 从pos_negs_scores第一维d的每个向量中从大到小获取分数最高的前k个数据
    #     pos_rank_idx = (rank < 1).nonzero()  # where hit 0, o is our target item#因为只有第一个是正样本，将除正样本以外的下标设为0
    #     if torch.any((rank < 1).sum(-1) > 1):  # sum(-1)表示tensor最后一维元素求和之后输出。如果>1说明正样本被推荐
    #         print("user embedding:", user_embedding[0])
    #         print("item embedding:", pos_negs_items[0])
    #         print("score:", pos_negs_interaction[0])
    #         print("rank", rank)
    #         print("pos_rank:", pos_rank_idx)
    #         np.save("pos_negs.npy", pos_negs_interaction.data.cpu().numpy())
    #         raise RuntimeError("compute rank error")
    #     have_hit_num = pos_rank_idx.shape[0]  # pos_rank_idx第一维数量，即所有击中正样本的维
    #     have_hit_rank = pos_rank_idx[:, 1].float()
    #     # DCG_ = 1.0 / torch.log2(have_hit_rank+2)
    #     # NDCG = DCG_ * torch.log2(torch.Tensor([2.0]).cuda())
    #     if have_hit_num > 0:
    #         batch_NDCG = 1 / torch.log2(have_hit_rank + 2)  # NDCG is equal to this format
    #         batch_NDCG = batch_NDCG.sum()
    #     else:
    #         batch_NDCG = 0
    #     Batch_hit = have_hit_num * 1.0
    #
    #     return Batch_hit, batch_NDCG, pos_rank_idx[:, 0]



    # def test(self):#meta-test
    #     for i in range(self.n_loop):
    #         # on support set
    #         for i_batch, (x1, x2, y, y0) in enumerate(self.user_data_loader):
    #             x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
    #             pred_y = self.save_model(x1, x2)
    #             loss = self.loss_fn(pred_y, y)
    #             self.optimizer.zero_grad()
    #             loss.backward()  # local theta updating
    #             torch.nn.utils.clip_grad_norm_(self.save_model.parameters(), 5.)
    #             self.optimizer.step()
    #
    #     # D.I.Y your calculation for the results
    #     q_pred_y = self.save_model(self.q_x1, self.q_x2)  # on query set
    '''
    根据计算结果计算ndcg指标
    :param top_k: top k items that need to return are ordered by save_model prediction. default: 10
    无meta,test直接输出指标
    
    :return
    batch_hit, batch_ndcg, hit_idx
    '''

    def test(self,rec_model, inputs_data, top_k=10):
        user = inputs_data[:, 0]  # x[:,n]表⽰在全部数组（维）中取第n个数
        all_items = inputs_data[:, 1:]  # all item.
        user_embedding = self.user_embedding(user)#one user
        all_items_embedding=self.item_emb(all_items)
        test_result=[]#each row has scorces of one pos inter and neg_num(99)  neg inter
        for item in all_items:
            _,_,pred_y=rec_model(user,item)
            test_result.append(pred_y)
        _, rank = torch.topk(test_result, top_k)
        pos_rank_idx = (rank < 1).nonzero()
        if torch.any((rank < 1).sum(-1) > 1):  # sum(-1)表示tensor最后一维元素求和之后输出。如果>1说明正样本被推荐
            print("user embedding:", user_embedding[0])
            print("item embedding:", all_items[0])
            print("score:", test_result[0])
            print("rank", rank)
            print("pos_rank:", pos_rank_idx)
            np.save("pos_negs.npy", test_result.data.cpu().numpy())
            raise RuntimeError("compute rank error")
        have_hit_num = pos_rank_idx.shape[0]  # pos_rank_idx第一维数量，即所有击中正样本的维
        have_hit_rank = pos_rank_idx[:, 1].float()
        # DCG_ = 1.0 / torch.log2(have_hit_rank+2)
        # NDCG = DCG_ * torch.log2(torch.Tensor([2.0]).cuda())
        if have_hit_num > 0:
            batch_NDCG = 1 / torch.log2(have_hit_rank + 2)  # NDCG is equal to this format
            batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0

        return Batch_hit, batch_NDCG, pos_rank_idx[:, 0]

        #return ndcg(test_result,top_k)




class BaseModel(torch.nn.Module):
    def __init__(self,rec_module,u_emb_module,i_emb_module,u_load,i_load):
        super(BaseModel, self).__init__()

        self.item_load=i_load
        self.user_load=u_load

        self.user_embedding = u_emb_module
        self.item_embedding = i_emb_module
        self.rec_model = rec_module   #推荐模型

    def forward(self, user, item, u_rep=None):
        pi = self.item_load(item)
        pu = self.user_load(user)
        eu, ei = self.user_embedding(pu), self.item_embedding(pi)
        rec_value = self.rec_model(eu,ei)
        return eu, ei, rec_value

    def get_weights(self):
        u_emb_params = get_params(self.user_embedding.parameters())
        i_emb_params = get_params(self.item_embedding.parameters())
        rec_params = get_params(self.rec_model.parameters())
        return u_emb_params, i_emb_params, rec_params

    def get_zero_weights(self):
        zeros_like_u_emb_params = get_zeros_like_params(self.user_embedding.parameters())
        zeros_like_i_emb_params = get_zeros_like_params(self.item_embedding.parameters())
        zeros_like_rec_params = get_zeros_like_params(self.rec_model.parameters())
        return zeros_like_u_emb_params, zeros_like_i_emb_params, zeros_like_rec_params

    def init_weights(self, u_emb_para, i_emb_para, rec_para):
        init_params(self.user_embedding.parameters(), u_emb_para)
        init_params(self.item_embedding.parameters(), i_emb_para)
        init_params(self.rec_model.parameters(), rec_para)

    def get_grad(self):
        u_grad = get_grad(self.user_embedding.parameters())
        i_grad = get_grad(self.item_embedding.parameters())
        r_grad = get_grad(self.rec_model.parameters())
        return u_grad, i_grad, r_grad

'''memory efficient personal meta-learner
'''
class Mepm(torch.nn.Module):
    def __init__(self, args, datasets):
        super(Mepm, self).__init__()
        self.global_lr = args.global_lr
        self.local_lr = args.local_lr  # global_parameter
        self.gama = args.gama
        self.alpha = args.alpha

        self.l1 = args.lambda_1
        self.l2 = args.lambda_2

        self.data_name = args.data_name
        self.use_cuda = args.use_cuda
        self.layer_num = args.layer_nums
        self.activate_func = args.activate_func
        self.embedding_dim = args.embedding_dim


        self.rec_model = Recommender_model(self.embedding_dim, self.layer_num, self.activate_func,
                                           self.data_name)  # emb_dim,layer_num,act_func,dataset_type='movielens'

        if self.data_name == 'movielens':
            self.item_load = ml_item(config)
            self.user_load = ml_user(config)
        elif self.data_name == 'yelp':
            self.item_load = yelp_item(config)
            self.user_load = yelp_user(config)
        else:
            self.item_load = amazon_item(config)
            self.user_load = amazon_user(config)

        self.item_emb = ItemEmbedding(self.layer_num, default_info[self.data_name]['i_in_dim'] * self.embedding_dim,
                                      self.embedding_dim, activation=activation_func('relu'))  # .to(self.device)
        self.user_emb = UserEmbedding(self.layer_num, default_info[self.data_name]['u_in_dim'] * self.embedding_dim,
                                      self.embedding_dim, activation=activation_func(
                'relu'))  # .to(self.device)  # n_layer, in_dim, embedding_dim, activation='sigmoid'

        self.basemodel = BaseModel(self.rec_model, self.user_emb, self.item_emb,
                                   self.user_load, self.item_load)

        self.K=args.K
        '''inital 0. it's regularization for only new data update '''
        self.last_grad = []  # 与basemodel的grad形状一致
        self.omega = []      # 与basemodel参数形状一致
        self.local_update_parameter = []
        for name, param in self.basemodel.named_parameters():  # input loading don't need train
            if 'load' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                self.local_update_parameter.append(name)
                self.omega.append(torch.zeros(param.size()))
                self.last_grad.append(torch.zeros(param.size()))
        #********************************************************************
        # self.enc_hid_dim, self.dec_hid_dim=args.enc_hid_dim,args.dec_hid_dim
        # self.attention=Attention2(self.enc_hid_dim, self.dec_hid_dim)
        #self.attention = MultiheadAttention(1,1)
        #self.linner_layer_local_lr=torch.nn.Linear(4, 1)
        #self.optimizer_loc_lr = torch.optim.Adam(self.attention.parameters(), lr=self.global_lr)

        #self.item_attention = item_Attention(self.embedding_dim, 1)
        #self.optimizer_ia = torch.optim.Adam(self.item_attention.parameters(), lr=self.global_lr)

        self.store_parameters()
        # 用于global_update
        self.optimizer = torch.optim.Adam(self.basemodel.parameters(), lr=self.global_lr,
                                          weight_decay=0)  # 更新basemodel，更新eta and local_lr在global_update中再定义Adam

        self.epoch_num = args.epochs
        self.batch_num = args.batch_num
        self.batch_size=args.batch_size
        self.dataset = datasets

        self.representation= {}  #u_id: representation
        # self.all_item = np.ones(0, dtype=np.long)
        # self.recall = []
        # self.ndcg = []
        # self.hit_new_user = []
        # self.hit_new_item = []
        # self.epochs = args.epochs
        # self.run_stage = 0
        # self.test_num = []
        # self.user_hit = None

        # self.new_user = torch.from_numpy(datasets.test_new_user).long().cuda()
        # self.new_item = torch.from_numpy(datasets.test_new_item).long().cuda()

    '''store the basemodel parameters and the regularization of omega and gradient of last time'''
    def store_parameters(self):
        self.keep_params_model = deepcopy(self.basemodel.state_dict())#meta model
        self.params_name = list(self.keep_params_model.keys())
        self.params_len = len(self.keep_params_model)
        self.task_params = OrderedDict()

    def get_grad(self, loss):
        u_emb_grad = torch.autograd.grad(loss, self.basemodel.user_embedding.parameters(), create_graph=True)
        i_emb_grad = torch.autograd.grad(loss, self.basemodel.item_embedding.parameters(), create_graph=True)
        rec_grad = torch.autograd.grad(loss, self.basemodel.rec_model.parameters(), create_graph=True)  # loss对elf.save_model.parameters()求导得到梯度。grad[i]对应parameter[i]
        grad = u_emb_grad + i_emb_grad + rec_grad
        return grad

    # '''get regularization and update the omega and last_grad'''
    # def update_regularization(self, grad):

    '''
    :param m: the interactions'num of training dataset
    loss_var, loss_exp最后再进行一层权重
    
    ？？？？？？？？？要不要加入local_lr一起
    '''
    def personal_local_lr(self,loss,grad,m,lr):
        #[loss,grad,m]作为一个sequence输入attention中
        #得到attention的输出之后，再经过一个线性层输出一个值，然后经过sigmod映射到[0,1]
        grad_mean = [torch.mean(g) for g in grad]
        grad_mean= torch.mean(torch.stack(grad_mean))
        a=torch.unsqueeze(torch.unsqueeze(loss,dim=0),dim=0)
        b = torch.unsqueeze(torch.unsqueeze(grad_mean, dim=0), dim=0)
        c=torch.stack([a,b])
        # 目标是得到衡量这三者的重要程度之后重新组合的向量  若是直接作为embedding的话，embedding_dim=1，否则通过linner层进行嵌入，不能用Embedding来嵌入，因为是float类型的数值
        #factors=torch.stack([torch.unsqueeze(loss,dim=0),torch.unsqueeze(grad_mean,dim=0),torch.tensor([float(m)])]) #,torch.tensor([m])
        factors=torch.stack([torch.unsqueeze(torch.unsqueeze(loss,dim=0),dim=0),torch.unsqueeze(torch.unsqueeze(grad_mean, dim=0), dim=0), torch.unsqueeze(torch.tensor([float(m)]),dim=0)]) #,torch.tensor([m])
        att_factors, attn_weights = self.attention(factors, factors, factors)

        #factors = torch.stack([loss, grad_mean, torch.tensor(float(m))])  # ,torch.tensor([m])
        # a=factors.size()
        # emb=torch.nn.Linear(1,32)
        # f=emb(factors)
        #factor_att=torch.mm(factors, att_score)

        # att_factors = torch.squeeze(att_factors)
        # lr = torch.tensor([lr])
        # lr_factors = torch.cat([att_factors, lr], dim=0)
        #
        # out = self.linner_layer_local_lr(lr_factors)

        return self.attention.get_personal_lr(att_factors,lr)



    def aggregation_item(self, items):
        items_emb=self.basemodel.item_embedding(self.basemodel.item_load(items))
        att_factors, attn_weights = self.attention(items_emb, items_emb, items_emb)

        #基于attention聚合items,items_emb作为输入




    def update_rep(self,  i_emb, u_id):#u_emb,
        i_emb=torch.unsqueeze(i_emb,dim=1)
        att_factors, attn_weights = self.item_attention(i_emb,i_emb,i_emb)
        i_emb = torch.squeeze(att_factors, dim=1)
        return

    '''
    meta-train:only use new data
    Local-update on new data. and to test the meta-save_model after global update
    然后将新用户加入历史用户中方便后续训练采样
    '''
    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update, u_id=None):  # 一个用户的support_set
        #lr = (1 - 1 / (1 + len(support_set_x) )) * self.local_lr
        #lr = lr.data.item()
        # if lr <= 0 or lr >= 0.1: lr = self.local_lr
        #u_rep=self.representation.get(u_id,None)
        u_emb, i_emb = None, None
        if self.representation.get(u_id, None):
            u_rep= self.representation[u_id]
        else:
            u_rep = None

        #lr=self.local_lr
        lr = (1 - 1 / (1 + len(support_set_y) )) * self.local_lr
        inter_num=len(support_set_y)
        loss=None
        grad=None
        for idx in range(num_local_update):
            if idx > 0:
                #setting task specific learning-rate based on [loss, grad, data_num]  怎么记录，
                #lr = self.gama * self.personal_local_lr(loss,grad,inter_num,lr).item() + (1 - self.gama) * lr   # 怎么更新

                self.basemodel.load_state_dict(self.task_params)
            param_for_local_update = []
            all_param = list(self.basemodel.state_dict().values()) # initial meta parameter

            u_emb, i_emb, pred_y = self.basemodel(support_set_x[0], support_set_x[1], u_rep=u_rep)

            loss = F.mse_loss(pred_y, support_set_y.view(-1, 1),reduction="none")  # 均方误差
            loss_mean = torch.mean(loss)
            loss_var = torch.var(loss)

            loss = F.mse_loss(pred_y, support_set_y.view(-1, 1))
            self.basemodel.zero_grad()
            grad = self.get_grad(loss)
            
            print("memory of gradient is", sys.getsizeof(grad))
            #grad=torch.autograd.grad(loss,self.basemodel.parameters())
            # loss.backward()

            #????????????????怎么更新attention,optimizer_loc_lr没有更新到
            # if idx>1:
            #     self.basemodel.zero_grad()
            #     temp=self.attention.state_dict()
            #     self.attention.zero_grad()
            #     loss.backward()
            #     self.optimizer_loc_lr.step()
            #     temp2 = self.attention.state_dict()
            m = 0
            for i in range(self.params_len):
                if self.params_name[i] in self.local_update_parameter:
                    self.task_params[self.params_name[i]] = all_param[i] - lr * grad[m]
                    m += 1
                else:
                    self.task_params[self.params_name[i]] = all_param[i]

        self.basemodel.load_state_dict(self.task_params)  # 训练之后的任务特定模型
        _, _, supp_set_y_pred = self.basemodel(support_set_x[0], support_set_x[1])
        _, _, query_set_y_pred = self.basemodel(query_set_x[0], query_set_x[1])  # 根据训练之后的模型在测试集上得到输出
        #eu,ei
        # 更新任务表示
        #a=torch.unsqueeze(support_set_x[0][0],dim=0)
        #u_emb=self.basemodel.user_embedding(self.basemodel.user_load(a))
        #self.update_rep( i_emb, u_id)

        self.basemodel.load_state_dict(self.keep_params_model)  # 回到初始meta-parameter
        return supp_set_y_pred, query_set_y_pred



    # meta-train
    # using loss on query-set based on user-save_model that fine-tuning by support_set to global update the meta save_model
    #正则项定义:last meta parameters and current task gradient(for update the meta save_model)
    def global_update(self, support_set_x, support_set_y, query_set_x, query_set_y, num_local_update, u_ids=None):
        #theta = [param for n, param in self.basemodel.named_parameters() if "load" not in n]
        #reg=self.alpha*torch.norm((theta-self.omega),p=2)**2 / 2 - self.last_grad + self.l1 * torch.norm(theta,p=1)
        model = deepcopy(self.basemodel.state_dict())
        losses_q=0
        for k in range(self.K): # how to add regularizition to update meta model
            #k次regularize
            self.basemodel.zero_grad()
            losses=self.get_batch_loss(support_set_x, support_set_y, query_set_x, query_set_y, num_local_update,u_ids)
            losses_q = torch.stack(losses).mean(0)
            
            losses_v = torch.stack(losses).var(0)
            adap_glr = (1-1/(1+losses_q/losses_v))*self.global_lr
            
            grad = self.get_grad(losses_q)  # expection of losses
            model_params = self.basemodel.state_dict()
            #theta = [param for n, param in model_params if "load" not in n]
            self.last_grad=list(self.last_grad)
            m = 0
            for i in range(self.params_len):
                if self.params_name[i] in self.local_update_parameter:
                    model_params[self.params_name[i]].requires_grad = True
                    self.omega[m]=self.omega[m].cuda()
                    self.last_grad[m]=self.last_grad[m].cuda()

                    #reg = self.alpha * torch.norm((model_params[self.params_name[i]] - self.omega[m]), p=2) ** 2 / 2 - self.last_grad[m]* model_params[self.params_name[i]]# + self.l1 * torch.norm(model_params[self.params_name[i]], p=1)
                    reg = self.alpha * torch.norm((model_params[self.params_name[i]] - self.omega[m]), p=2) ** 2 / 2 - torch.dot(torch.flatten(self.last_grad[m]),torch.flatten(model_params[self.params_name[i]]))
                    reg_grad=torch.autograd.grad(reg,model_params[self.params_name[i]])#,torch.ones_like(reg))
                    #p=deepcopy(model_params[self.params_name[i]] )
                    # model_params[self.params_name[i]] = model_params[self.params_name[i]] - self.global_lr * (grad[m]+reg_grad[0]  + self.l1 * torch.norm(model_params[self.params_name[i]], p=1))
                    model_params[self.params_name[i]] = model_params[self.params_name[i]] - adap_glr * (grad[m]+reg_grad[0]  + self.l1 * torch.norm(model_params[self.params_name[i]], p=1))
                    m+=1
            if k==self.K-1:
                for i in range(len(grad)):
                    self.last_grad[i]=(grad[i]+self.last_grad[i])/2
                # losses=torch.stack(losses)
                # for i in range(len(losses)):
                #     TensorWriter.add_scalar('loss_{}'.format(stage_id), losses[i],i)
                print(torch.stack(losses))
            self.basemodel.load_state_dict(model_params)
            self.store_parameters()
        temp=self.basemodel.state_dict()
        model_params=[param for n,param in self.basemodel.named_parameters() if "load" not in n]
        self.update_reg(model_params)
        return losses_q

    def update_reg(self,m_params):
        for i in range(len(m_params)):
            self.omega[i] = (self.omega[i] + m_params[i] - 1/self.alpha * self.last_grad[i]) / 2


    def get_batch_loss(self,support_set_x, support_set_y, query_set_x, query_set_y, num_local_update, u_ids):
        batch_sz = len(support_set_y)
        losses_q = []

        if self.use_cuda:
            for i in range(batch_sz):
                support_set_x[i] = [support_set_x[i][0].cuda(), support_set_x[i][1].cuda()]
                support_set_y[i] = support_set_y[i].cuda()
                query_set_x[i] = [query_set_x[i][0].cuda(), query_set_x[i][1].cuda()]
                query_set_y[i] = query_set_y[i].cuda()
        for i in range(batch_sz):  # meta-train on each task(user)
            self.basemodel.zero_grad()

            supp_set_y_pred, query_set_y_pred = self.forward(support_set_x[i], support_set_y[i], query_set_x[i], num_local_update, u_ids[i])

            # # 更新任务表示
            # self.update_rep(support_set_x[i], support_set_y[i], query_set_x[i], query_set_y[i], u_ids[i])

            loss_q = F.mse_loss(query_set_y_pred, query_set_y[i].view(-1, 1))  # reshape one col
            # if loss_s.item() >= self.C: continue
            losses_q.append(loss_q)

        return losses_q

    def k_reg(self):

        for i in range(self.K):
            model_params = deepcopy(self.basemodel.state_dict())
            theta = [param for n, param in model_params if "load" not in n]
            reg = self.alpha * torch.norm((theta - self.omega), p=2) ** 2 / 2 - grad+ self.l1 * torch.norm(
                theta, p=1)
            m=0
            for j in range(self.params_len):
                if self.params_name[i] in self.local_update_parameter:
                    model_params[self.params_name[i]]=model_params[self.params_name[i]]-self.global_lr * ()
            self.store_parameters()



def get_parse():
    parser = argparse.ArgumentParser(description='mepml parameters.')
    parser.add_argument('--layer_nums', type=int, default=3,  #
                        help='number of fc layers.')
    parser.add_argument('--global_lr', type=float, default=1e-3,  # 0.01
                        help='Global learning rate gamma for global updating.')
    parser.add_argument('--local_lr', type=float, default=1e-5,
                        help='global parameter alpha_hat for local user save_model updating')
    parser.add_argument('--gama', type=float, default=1e-4,
                        help='hyper-parameter for calculate the task specific local_lr')
    parser.add_argument('--alpha', type=float, default=1e-5,
                        help='hyper-parameter for regularization')
    parser.add_argument('--K', type=int, default=7,  #3，7，10，13
                        help='K corrected gradient descent steps.')

    parser.add_argument('--lambda_1', type=float, default=1e-7,
                        help='parameter for L1 regularization')
    parser.add_argument('--lambda_2', type=float, default=1e-7,
                        help='parameter for L2 regularization')
    parser.add_argument('--tao', type=float, default=1e-7,
                        help='parameter for calculate the eta')

    parser.add_argument('--epochs', type=int, default=20,  #
                        help='Number of epochs to train of each stage.')
    parser.add_argument('--batch_num', type=int, default=32,  # full-retrain 256 , others 64
                        help='the number of batches in each epoch.')


    parser.add_argument('--batch_size', type=int, default=32,  # full-retrain 256 , others 64
                        help='batch size of train.')
    parser.add_argument('--use_cuda', type=bool, default=True, help="")

    parser.add_argument('--embedding_dim', type=int, default=32,  # 64
                        help='dim of embedding.')
    parser.add_argument('--activate_func', type=str, default="relu",  #
                        help='activation function')
    parser.add_argument('--cuda', type=int, default=1,
                        help='which GPU be used?.default 1')

    parser.add_argument('--data_path', default='/home/ly525999/MEPM-MASTER/datasets/',
                        help='data path')
    parser.add_argument('--data_name', default='movielens',
                        help='dataset name')
    # parser.add_argument('--start_idx', type=int, default=30,
    #                     help='retraining from which period: yelp 30, news(adressa) 48')
    return parser


'''
    数据集
    movielens
    yelp
    amazon
'''
if __name__ == "__main__":
    master_path = "./datasets/movielens"
    parser = get_parse()
    args = parser.parse_args()

    dataset = online_movielens_1m()
    # if not os.path.exists("{}/".format(master_path)):
    #     os.mkdir("{}/".format(master_path))
    #     # preparing dataset. It needs about 22GB of your hard disk space.
    #     generate(master_path)

    # training save_model
    mepm = Mepm(args, dataset)
    
    # cold_test_dataset, _ = dataset.get_testDataset(master_path)
    # model_filename = "{}/models.pkl".format(master_path)
    stage_id = 0  # 针对每个stage都做epoch-batch训练
    while stage_id < 45:  # 50
        print("*****************start training of stage", stage_id)
        memory=0
        model_filename = "./save_model2/models_{}.pkl".format(stage_id)
        regularization_name = "./save_model2/reg_{}.pkl".format(stage_id)

        rep_name="./save_model2/rep.pkl"  # save the tasks' representation
        if stage_id and os.path.exists("./save_model2/models_{}.pkl".format(stage_id - 1)):
            trained_state_dict = torch.load("./save_model2/models_{}.pkl".format(stage_id - 1))
            mepm.load_state_dict(trained_state_dict)
            reg = torch.load("./save_model2/reg_{}.pkl".format(stage_id - 1))
            mepm.omega = reg['omega']
            mepm.last_grad = reg["last_grad"]

        #load the tasks' representation
        

        # test_num=dataset.get_test_num(stage_id)
        # Load training dataset. users_id is all users'id in this stage used to get representation from self.representation
        current_dataset, users_id = dataset.next_dataset(stage_id, master_path, types="only_new")#types="only_new")
        memory+=binary_conversion(sys.getsizeof(current_dataset))
        cold_test_dataset, _ = dataset.next_dataset(stage_id, master_path, types="test_cold")

        #his_test_dataset=dataset.next_dataset(stage_id-1, master_path, types="only_new")

        # ndcg5, hr5, c_ndcg5, c_hr5, ndcg10, hr10, c_ndcg10, c_hr10 = training(mepm, current_dataset, cold_test_dataset,
        #                                                                       epoch_num=args.epochs,
        #                                                                       batch_size=args.batch_size,
        #                                                                       stage_id=stage_id, model_save=True,
        #                                                                       model_filename=model_filename,reg_name=regularization_name)#,his_dataset=his_test_dataset)  # ,test_dataset)
        # TensorWriter.add_scalars('stage_ndcg@5', {'query-set': ndcg5, 'cold-user': c_ndcg5}, stage_id)
        # TensorWriter.add_scalars('stage_hr@5', {'query-set': hr5, 'cold-user': c_hr5}, stage_id)
        #
        # TensorWriter.add_scalars('stage_ndcg@10', {'query-set': ndcg10, 'cold-user': c_ndcg10}, stage_id)
        # TensorWriter.add_scalars('stage_hr@10', {'query-set': hr10, 'cold-user': c_hr10}, stage_id)
        ndcg1, c_ndcg1, ndcg3, c_ndcg3, mae, c_mae = training(mepm, current_dataset, cold_test_dataset,
                                                              epoch_num=args.epochs,
                                                              batch_size=args.batch_size,
                                                              stage_id=stage_id, model_save=True,
                                                              model_filename=model_filename,
                                                              reg_name=regularization_name, users_id=users_id, rep_name=rep_name)  # ,his_dataset=his_test_dataset)  # ,test_dataset)
        TensorWriter.add_scalars('stage_ndcg@1', {'query-set': ndcg1, 'cold-user': c_ndcg1}, stage_id)
        TensorWriter.add_scalars('stage_ndcg@3', {'query-set': ndcg3, 'cold-user': c_ndcg3}, stage_id)
        TensorWriter.add_scalars('stage_mae', {'query-set': mae, 'cold-user': c_mae}, stage_id)
        TensorWriter.add_scalar('stage_memory', memory, stage_id)
        print("------------------finish stage {} training! The memory cost is {} KB---------------".format(stage_id,memory))

        # test save_model
        # model_filename = "{}/stage_model/models_{}.pkl".format(master_path, stage_id)
        # trained_state_dict = torch.load(model_filename)
        # form.load_state_dict(trained_state_dict)
        # ndcg, hr, c_ndcg, c_hr =test(form,current_dataset,cold_test_dataset,epoch_num=args.epochs,batch_num=args.batch_num, stage_id=stage_id)
        # TensorWriter2.add_scalars('stage_ndcg@5', {'query-set':ndcg,'cold-user':c_ndcg}, stage_id)
        # TensorWriter2.add_scalars('stage_hr@5', {'query-set':hr,'cold-user':c_hr}, stage_id)

        # dataset.add_new_user_to_history(stage_id)#add new users to history buffer
        stage_id += 1

        # else:
        #     trained_state_dict = torch.load(model_filename)
        #     form.load_state_dict(trained_state_dict)

'''global update'''


def meta_test(self, model, new_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    pass


