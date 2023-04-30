import math
from functools import total_ordering
from math import floor
from tkinter import E
import numpy as np
import torch
import shutil

try:
    from tqdm import tqdm
except:
    pass
from torch.utils.data import Dataset
import copy
import datetime
import pandas as pd
import re
import os

import random
import pickle

import collections
from copy import deepcopy

# from config import states

import sys


def binary_conversion(var: int):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    assert isinstance(var, int)
    # if var <= 1024:
    return round(var / 1024, 2)


'''
dataset=online_movielens_1m(stage_num)
to generate next streaming support set and query set 
self.get_streaming_interAndY(stage_id=,dataset.u_dict,dtaset.m_dict)
to get stageid support and query set with open file
movielens/support/supp_x_stageid_第几个用户.pkl
'''


class online_movielens_1m(object):
    def __init__(self, stage_num=50):
        self.path = './datasets/movielens'
        self.user_data, self.item_data, self.score_data = self.load()
        self.user_num, self.item_num = len(self.user_data) - 1, len(self.item_data)

        self.u_dict, self.m_dict = self.generate(self.path, self.item_data, self.user_data)
        # warm_users,cold_users,warm_users_data,cold_users_data=self.split_warm_cold(self.score_data)
        #
        # # self.generate_streaming(stage_num,warm_users_data)
        # # self.generate_streaming(stage_num,self.score_data)
        # #
        # # #generate each time(stage) support and query set
        # each_stage_cu_num=len(cold_users)//stage_num
        # cu_inter,cu_inter_y=self.get_cold_inter_y(cold_users_data)
        # for i in tqdm(range(stage_num)):
        #     self.generate_current_suppAndQuery(i,types="not_only_new")
        #     if i == stage_num-1:
        #         stage_cu_id=cold_users[each_stage_cu_num*i:]
        #     else:
        #         stage_cu_id=cold_users[each_stage_cu_num*i:each_stage_cu_num*(i+1)]
        #     self.generate_cold_dataset(stage_cu_id,cu_inter,cu_inter_y,i)

        # for state in states:
        #     self.generate_streaming(self.score_data,stage_num)#,'./movielens',state)

    # load all users and items and rating in row data
    def load(self):

        path = "./datasets/movielens/ml-1m"
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)

        profile_data = pd.read_csv(profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
                                   sep="::", engine='python')
        item_data = pd.read_csv(
            item_data_path,
            names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot',
                   'poster'],
            sep="::", engine='python', encoding="utf-8"
        )
        item_data = item_data.drop(columns=['released', 'writer', 'actors', 'plot', 'poster'])

        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )
        score_data = score_data.sort_values(by=['timestamp'])  # 按时间戳升序

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data

    # split score data by timestamp as online data
    '''
    :param load_dataset:movielens_1m.load()
    train and test:  80:20
    leave-one-out evaluation 
    写入数据集
    先划分stage_num个数据片，然后划分每个数据片（8:2）为train and test

    划分用户的interaction为train-set and test-set
    warm-user and cold-user
    '''

    def split_warm_cold(self, rat_data, max_count=10):
        storing_path = './datasets'
        dataset = 'movielens'
        # 按照时间划分数据集
        sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, split_time, end_time = sorted_time['time'][0], sorted_time['time'][
            round(0.8 * len(rat_data))], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        print('start time %s, split_time %s, end time %s' % (start_time, split_time, end_time))
        user_warm_list, user_cold_list, user_counts = [], [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()
        n_user_ids = rat_data.user_id.nunique()

        warm, cold = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'time']), pd.DataFrame(
            columns=['user_id', 'movie_id', 'rating', 'time'])
        for u_id in tqdm(user_ids):
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)
            if u_count >= max_count and u_count < 1000:  # 过滤数据少于max_count的用户
                new_u_info = u_info.iloc[:max_count, :]
                new_df = new_df.append(new_u_info, ignore_index=True)
                u_time = u_info['time'][0]
                if u_time < split_time:
                    user_warm_list.append(u_id)
                    warm = pd.concat([warm, u_info]).reset_index(drop=True)
                else:  # choose the user as codl user which have the number of interacting items less than 100 after split_time
                    if u_count < 50:
                        user_cold_list.append(u_id)
                        cold = pd.concat([cold, u_info]).reset_index(drop=True)
                    else:
                        user_warm_list.append(u_id)
                        warm = pd.concat([warm, u_info]).reset_index(drop=True)
            user_counts.append(u_count)
        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('min count: %d, avg count: %d, max count: %d' % (
        min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
                          'user_cold_ids': user_cold_list}
        pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
        return user_warm_list, user_cold_list, warm, cold  # pd.DataFrame(warm,columns=['user_id', 'movie_id', 'rating', 'time']),pd.DataFrame(cold,columns=['user_id', 'movie_id', 'rating', 'time'],)

    # 对warm_user的数据生成streaming data
    def generate_streaming(self, stage_num, rate_d):
        sorted_time = rate_d.sort_values(by='time', ascending=True).reset_index(drop=True)
        stage_data_num = len(rate_d) // stage_num
        for i in tqdm(range(stage_num)):
            stage_data = sorted_time.iloc[i * stage_data_num:(i + 1) * stage_data_num, :]
            stage_data = pd.DataFrame(stage_data)
            stage_data.to_csv(self.path + '/streaming/' + str(i) + '.dat', index=False)

    '''
    based on choice of types to determine only current streaming or current and previous streaming to load dataset
    return current_all_users,current_all_inter(u_id and m_id),current_all_inter_rating(rate of u_id and m_id)
    '''

    def get_next_dataset(self, stage_id, types="not_only_new"):
        current_all_users, current_all_items, current_all_inter = [], set(), []
        try:
            # cold_users={}#leave-one-out:选择给定时间之后交互较少的用户作为cold-user，用于test-on-cold-users
            # current_stage_users=[]
            if types == "not_only_new":  # 将之前所有streaming的数据整合到一起作为历史数据
                for i in range(0, stage_id + 1):  # 将当前stage之前的数据
                    score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                    score_data = pd.read_csv(score_data_path, names=['user_id', 'movie_id', 'rating', 'time'],
                                             engine='python')
                    for idx, row in score_data.iterrows():
                        if row['user_id'] == 'user_id': continue
                        if row['user_id'] not in current_all_users:
                            current_all_users.append(row['user_id'])  # 保持时间顺序
                        current_all_items.add(row['movie_id'])
                        current_all_inter.append([row['user_id'], row['movie_id'], row['rating'], row['time']])
                        # current_all_inter=current_all_inter.append(row).reset_index(drop=True)
                        # if i==stage_id:
                        #     current_stage_users.append(row['user_id'])
                # current_stage_users = sorted(set(current_all_users),key=current_all_users.index)
            else:
                score_data_path = self.path + '/streaming/' + str(stage_id) + '.dat'
                score_data = pd.read_csv(score_data_path, names=['user_id', 'movie_id', 'rating', 'time'],
                                         engine='python')
                for idx, row in score_data.iterrows():
                    if row['user_id'] == 'user_id':
                        continue
                    if row['user_id'] not in current_all_users:
                        current_all_users.append(row['user_id'])
                    current_all_items.add(row['movie_id'])
                    current_all_inter.append([row['user_id'], row['movie_id'], float(row['rating']), row['time']])
                    # current_all_inter=current_all_inter.append(row).reset_index(drop=True)
                # current_all_users = sorted(set(current_all_users),key=current_all_users.index)#去重不改变顺序
                # current_stage_users=deepcopy(current_all_users)

            users_movie_inter = collections.defaultdict(list)
            users_movie_inter_y = collections.defaultdict(list)
            for row in current_all_inter:  # 因为stage_data中的数据是按时间排序的
                # if row[-2]=='rating':continue#跳过表头
                users_movie_inter[row[0]].append(row[1])  # 直接取前80%就是时间上前80%的交互
                users_movie_inter_y[row[0]].append(row[-2])

                # 选择在给定时间之后交互很少的用户作为cold-users（选择在当前streaming中交互很少的用户作为冷启动用户）
            return current_all_users, users_movie_inter, users_movie_inter_y
        except:
            print("read stage data roung , may be there is no new data,finished")
            return None, None, None

    '''分场景，将每个streaming data中的users划分为support-set and query-set
    先对每个streaming data中的用户看过的电影，以及评分y

    随机选择前80%的交互作为supprot set，其余20%交互为query-set
    '''

    # not only new
    # test and val:choose the latest items as the test and validation instance following the leave-one-out evaluation
    # 根据时间预先划分streaming data
    def generate_current_suppAndQuery(self, stage_id, types="not_only_new"):
        # 基于get_next的数据集采样用户，生成这些用户的support-set and query-set
        current_all_users, users_movie_inter, users_movie_inter_y = self.get_next_dataset(stage_id)
        if types == "not_only_new":
            dir = "stage_dataset"
        else:
            dir = "stage_dataset_only_new"
        if users_movie_inter == None:
            print("***********read stage data roung , may be there is no new data,finished***************")
            return None

        users = []  # 从current_all_users中采样用户  还是全部用户？？？？？？？？？？？？？？？？？？？？
        users = current_all_users

        n = len(users)
        # generate the support set and query set
        idx = 0
        for u_id in users:  # 有时间顺序的
            support_x = []
            query_x = []
            seen_movie = len(users_movie_inter[u_id])
            indices = list(range(seen_movie))
            if seen_movie < 10 or seen_movie > 1000:
                continue
            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set
            # for idx in indices[:floor(seen_movie*0.8)]:
            #     support_x.append(users_movie_inter[u_id])
            # for idx in indices[floor(seen_movie*0.8):]:
            #     query_x.append(users_movie_inter[u_id])
            tmp_x = np.array(users_movie_inter[u_id])
            tmp_y = np.array(users_movie_inter_y[u_id])

            s_user = pd.DataFrame([u_id] * floor(seen_movie * 0.8), columns=['user_id'])
            q_user = pd.DataFrame([u_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

            support_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['movie_id'])
            query_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['movie_id'])
            support_x = s_user.join(support_x)
            query_x = q_user.join(query_x)
            support_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
            query_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            folder_path = "{}/{}/{}".format(self.path, dir, stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            support_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['user_id', 'movie_id'], index=False)
            support_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['rating'], index=False)
            query_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['user_id', 'movie_id'], index=False)
            query_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['rating'], index=False)
            idx += 1
        print("**********generate dataset successfully******************")
        return True
        # 是写入文件还是直接返回suppor and query

    def get_cold_inter_y(self, cold_users_data):
        cu_movie_inter = collections.defaultdict(list)
        cu_movie_inter_y = collections.defaultdict(list)
        # each_stage_num=len(cold_user_idx)//stage_num
        for idx, row in cold_users_data.iterrows():  # 因为stage_data中的数据是按时间排序的
            cu_movie_inter[row['user_id']].append(row['movie_id'])  # 直接取前80%就是时间上前80%的交互
            cu_movie_inter_y[row['user_id']].append(row['rating'])
        return cu_movie_inter, cu_movie_inter_y

    # 直接对每个cold
    def generate_cold_dataset(self, cold_user_idx, cu_movie_inter, cu_movie_inter_y, stage_id):
        idx = 0
        for cold_id in cold_user_idx:

            seen_movie = len(cu_movie_inter[cold_id])
            indices = list(range(seen_movie))

            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set
            tmp_x = np.array(cu_movie_inter[cold_id])
            tmp_y = np.array(cu_movie_inter_y[cold_id])
            train_user = pd.DataFrame([cold_id] * floor(seen_movie * 0.8), columns=['user_id'])
            test_user = pd.DataFrame([cold_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

            train_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['movie_id'])
            train_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
            train_x = train_user.join(train_x)

            test_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['movie_id'])
            test_x = test_user.join(test_x)
            test_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            folder_path = "{}/{}/{}".format(self.path, 'cold_user', stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            train_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['user_id', 'movie_id'], index=False)
            train_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['rating'], index=False)
            test_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['user_id', 'movie_id'], index=False)
            test_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['rating'], index=False)
            idx += 1

    def user_converting(self, row, age_list, gender_list, occupation_list, zip_list):
        # gender_dim: 2, age_dim: 7, occupation: 21
        gender_idx = gender_list.index(str(row['gender']))
        age_idx = age_list.index(str(row['age']))
        occupation_idx = occupation_list.index(str(row['occupation_code']))
        zip_idx = zip_list.index(str(row['zip'])[:5])

        return [gender_idx, age_idx, occupation_idx, zip_idx]

    def item_converting(self, row, rate_list, genre_list, director_list, year_list):
        # rate_dim: 6, year_dim: 1,  genre_dim:25, director_dim: 2186,
        rate_idx = rate_list.index(str(row['rate']))
        genre_idx = [0] * 25
        for genre in str(row['genre']).split(", "):
            idx = genre_list.index(genre)
            genre_idx[idx] = 1
        director_idx = [0] * 2186
        for director in str(row['director']).split(", "):
            idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
            director_idx[idx] = 1
        year_idx = year_list.index(row['year'])
        out_list = list([rate_idx, year_idx])
        out_list.extend(genre_idx)
        out_list.extend(director_idx)
        return out_list

    # 将item信息转换为index格式（可以看作one-hot）
    # 用input_loading
    # def item_converting(self,row, rate_list, genre_list, director_list, actor_list):
    #     rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    #     genre_idx = torch.zeros(1, 25).long()
    #     for genre in str(row['genre']).split(", "):
    #         idx = genre_list.index(genre)
    #         genre_idx[0, idx] = 1
    #     director_idx = torch.zeros(1, 2186).long()
    #     for director in str(row['director']).split(", "):
    #         idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
    #         director_idx[0, idx] = 1
    #     actor_idx = torch.zeros(1, 8030).long()
    #     for actor in str(row['actors']).split(", "):
    #         idx = actor_list.index(actor)
    #         actor_idx[0, idx] = 1
    #     return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)
    ##将user信息转换为index格式（可以看作one-hot） 
    # def user_converting(self,row, gender_list, age_list, occupation_list, zipcode_list):
    #     gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    #     age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    #     occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    #     zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    #     return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)

    def load_list(self, fname):
        list_ = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():
                list_.append(line.strip())
        return list_

    '''dataset = movielens_1m()
    item_data=dataset.item_data
    user_data=dataset.user_data
    '''

    def generate(self, master_path, item_data, user_data):

        rate_list = self.load_list("{}/{}/m_rate.txt".format(self.path, 'ml-1m'))
        genre_list = self.load_list("{}/{}/m_genre.txt".format(self.path, 'ml-1m'))
        # actor_list = self.load_list("{}/{}/m_actor.txt".format(self.path,'ml-1m'))
        director_list = self.load_list("{}/{}/m_director.txt".format(self.path, 'ml-1m'))
        gender_list = self.load_list("{}/{}/m_gender.txt".format(self.path, 'ml-1m'))
        age_list = self.load_list("{}/{}/m_age.txt".format(self.path, 'ml-1m'))
        occupation_list = self.load_list("{}/{}/m_occupation.txt".format(self.path, 'ml-1m'))
        zipcode_list = self.load_list("{}/{}/m_zipcode.txt".format(self.path, 'ml-1m'))
        year_list = list(item_data.year.unique())

        # dataset = movielens_1m()
        # hashmap for item information
        if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
            movie_dict = collections.OrderedDict()
            for idx, row in item_data.iterrows():
                if row['movie_id'] == 'movie_id': continue  # 跳过表头
                m_info = self.item_converting(row, rate_list, genre_list, director_list, year_list)
                movie_dict[row['movie_id']] = m_info
            pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            user_dict = collections.OrderedDict()
            for idx, row in user_data.iterrows():
                if row['user_id'] == 'user_id' or row['gender'] == 'Gender': continue  # 跳过表头
                u_info = self.user_converting(row, age_list, gender_list, occupation_list, zipcode_list)
                user_dict[int(row['user_id'])] = u_info
            pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
        else:
            user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))
        return user_dict, movie_dict

    '''打开下一时刻的streaming data
    movielens/support/supp_x_stageid_{}.pkl    movielens/support/supp_y_stageid_{}.pkl
    movielens/query/query_x_stageid_{}.pkl     movielens/query/query_y_stageid_{}.pkl 
    x_stageid_{}.pkl  y_stageid_{}.pkl
    D_test=D_val
    一个stage_id对应一个streaming
    return: total_dataset
    根据dataset_path来生成warm and cold的数据集
    warm:datasets/movielens/stage_dataset        cold:datasets/movielens/cold_user
    '''

    # 对数据集采样，选择哪些用户来更新meta-model
    def next_sampling_tasks(self, sample_num, stage_id, dataset_path, types="not_only_new", neg_num=None):
        if types == "not_only_new":
            dir = "/stage_dataset/"
        elif types == "test_cold":
            dir = "/cold_user/"
        else:
            dir = "/stage_dataset_only_new/"
        dataset_path = dataset_path + dir + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
            return None
        all_tasks = [idx for idx in range(len(data_file) // 4)]
        random.shuffle(all_tasks)
        sampled_tasks = all_tasks[:sample_num]
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        neg_x_s = []
        neg_y_s = []
        users_id = []
        for idx in sampled_tasks:
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            u_id = None
            if len(s_x["user_id"]) < 1: continue  # 过滤空文件
            inter_item = 0
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                u_id = int(row['user_id'])
                inter_item += 1
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    a = self.u_dict[row['user_id']]
                    support_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.m_dict[row['movie_id']]])
                # s_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
                # except:
                #     support_x= s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted
            if u_id:
                users_id.append(u_id)

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            for idx, row in q_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.m_dict[row['movie_id']]])

                # q_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     query_x= torch.cat((query_x, q_x_converted), 0)#按维数0（行）拼接
                # except:
                #     query_x= q_x_converted
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

            if neg_num:
                neg_inter, neg_inter_y = self.gen_neg_train(s_x, q_x,
                                                            neg_number=inter_item)  # the number of neg_item is equal to pos_item number
                neg_x = [None]
                neg_y = [None]
                for idx, row in neg_inter.iterrows():
                    if row['movie_id'] == "movie_id": continue
                    # try:
                    #     neg_x[0] = torch.cat((neg_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])), 0)  # 按维数0（行）拼接
                    # except:
                    #     neg_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                    # try:
                    #     neg_x[1] = torch.cat((neg_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                    #                              0)  # 按维数0（行）拼接
                    # except:
                    #     neg_x[1] = torch.tensor([self.m_dict[row['movie_id']]])

                    try:
                        neg_x = torch.cat((neg_x, torch.tensor([self.m_dict[row['movie_id']]])), 0)  # 按维数0（行）拼接
                    except:
                        neg_x = torch.tensor([self.m_dict[row['movie_id']]])
                    s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
                    try:
                        neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
                    except:
                        neg_y = s_y_converted
                neg_x_s.append(neg_x)
                neg_y_s.append(neg_y)

        if neg_num:
            d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
        else:
            d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t, users_id

    def next_dataset(self, stage_id, dataset_path, types="not_only_new", neg_num=0):
        if types == "not_only_new":
            dir = "/stage_dataset/"
        elif types == "test_cold":
            dir = "/cold_user/"
        else:
            dir = "/stage_dataset_only_new/"
        dataset_path = dataset_path + dir + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
            return None
        interactions = 0
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        neg_x_s = []
        neg_y_s = []
        users_id = []
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            u_id = None
            if len(s_x["user_id"]) < 1: continue  # 过滤空文件
            inter_item = 0
            interactions += len(s_x)
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                u_id = row['user_id']
                inter_item += 1
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    a = self.u_dict[row['user_id']]
                    support_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.m_dict[row['movie_id']]])
                # s_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
                # except:
                #     support_x= s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted
            if u_id:
                users_id.append(u_id)

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            interactions += len(q_x)
            for idx, row in q_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['movie_id']]])), 0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.m_dict[row['movie_id']]])

                # q_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     query_x= torch.cat((query_x, q_x_converted), 0)#按维数0（行）拼接
                # except:
                #     query_x= q_x_converted
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

            if neg_num:
                neg_inter, neg_inter_y = self.gen_neg_train(s_x, q_x,
                                                            neg_number=inter_item)  # the number of neg_item is equal to pos_item number
                neg_x = [None]
                neg_y = [None]
                for idx, row in neg_inter.iterrows():
                    if row['movie_id'] == "movie_id": continue
                    # try:
                    #     neg_x[0] = torch.cat((neg_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])), 0)  # 按维数0（行）拼接
                    # except:
                    #     neg_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                    # try:
                    #     neg_x[1] = torch.cat((neg_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                    #                              0)  # 按维数0（行）拼接
                    # except:
                    #     neg_x[1] = torch.tensor([self.m_dict[row['movie_id']]])

                    try:
                        neg_x = torch.cat((neg_x, torch.tensor([self.m_dict[row['movie_id']]])), 0)  # 按维数0（行）拼接
                    except:
                        neg_x = torch.tensor([self.m_dict[row['movie_id']]])
                    s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
                    try:
                        neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
                    except:
                        neg_y = s_y_converted
                neg_x_s.append(neg_x)
                neg_y_s.append(neg_y)

        if neg_num:
            d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
        else:
            d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))

        print("{} interactions:{}".format(types, interactions))
        return d_t, users_id

    def get_testDataset(self, dataset_path, neg_num=0):
        dataset_path = dataset_path + "/cold_user"
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
            return None
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        neg_x_s = []
        neg_y_s = []
        users_id = []
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            u_id = None
            if len(s_x["user_id"]) < 1: continue  # 过滤空文件
            inter_item = 0
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                u_id = int(row['user_id'])
                inter_item += 1
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    a = self.u_dict[row['user_id']]
                    support_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.m_dict[row['movie_id']]])
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted
            if u_id:
                users_id.append(u_id)

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            for idx, row in q_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['movie_id']]])), 0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.m_dict[row['movie_id']]])

                # q_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     query_x= torch.cat((query_x, q_x_converted), 0)#按维数0（行）拼接
                # except:
                #     query_x= q_x_converted
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

            if neg_num:
                neg_inter, neg_inter_y = self.gen_neg_train(s_x, q_x,
                                                            neg_number=inter_item)  # the number of neg_item is equal to pos_item number
                neg_x = [None]
                neg_y = [None]
                for idx, row in neg_inter.iterrows():
                    if row['movie_id'] == "movie_id": continue

                    try:
                        neg_x = torch.cat((neg_x, torch.tensor([self.m_dict[row['movie_id']]])), 0)  # 按维数0（行）拼接
                    except:
                        neg_x = torch.tensor([self.m_dict[row['movie_id']]])
                    s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
                    try:
                        neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
                    except:
                        neg_y = s_y_converted
                neg_x_s.append(neg_x)
                neg_y_s.append(neg_y)

        if neg_num:
            d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
        else:
            d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t, users_id

    # generate cold-user's train set and test-set based on leave-one-out
    # data_path: ./movielens/cold-user/stage_id/
    # test_num范围是一个data_x的长度，或者取最小的
    # 在后续模型test-on-cold-user的时候读取文件，根据leave-one-out来从meta-save_model finetuning and test(validation)。
    # def next_test_cold_user(self,stage_id):
    #     try:
    #         file=os.listdir(self.path+"/cold_user/{}".format(stage_id))
    #     except:
    #         print("no file!")
    #         return False
    #     train_xs,train_ys=[],[]
    #     test_xs,test_ys=[],[]
    #     for idx in range(len(file)//4):
    #         train_x_path=self.path+"/cold_user/{}/supp_x_{}.dat".format(stage_id,idx)
    #         train_y_path=self.path+"/cold_user/{}/supp_y_{}.dat".format(stage_id,idx)
    #         test_x_path=self.path+"/cold_user/{}/query_x_{}.dat".format(stage_id,idx)
    #         test_y_path=self.path+"/cold_user/{}/query_y_{}.dat".format(stage_id,idx)
    #         try:
    #             x = pd.read_csv(x_path, names=['user_id', 'movid_id'], engine='python')
    #             y = pd.read_csv(y_path, names=['movid_id'], engine='python')
    #         except:
    #             print("filename error or filepath error!")
    #             return

    #         for test_num in range(x.shape[0]-1):
    #             train_x=None
    #             test_x=None
    #             train_y=None
    #             test_y=None
    #             for idx,row in x.iterrows():
    #                 if idx==test_num:
    #                     test_x=torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[row['user_id']]), 1)
    #                     test_y=self.m_dict[int(y.iloc[idx]['movie_id'])]
    #                 else:
    #                     train_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[row['user_id']]), 1)#按维数1（列）拼接
    #                     train_y_converted = self.m_dict[int(y.iloc[idx]['movie_id'])]
    #                     try:
    #                         train_x= torch.cat((train_x, train_x_converted), 0)#按维数0（行）拼接
    #                     except:
    #                         train_x= train_x_converted
    #                     try:
    #                         train_y= torch.cat((train_y, train_y_converted), 0)#按维数0（行）拼接
    #                     except:
    #                         train_y= train_y_converted
    #             train_xs.append(train_x)
    #             train_ys.append(train_y)
    #             test_xs.append(test_x)
    #             test_ys.append(test_y)
    #     cold_user_data = list(zip(train_xs,train_ys,test_xs,test_ys))#total_dataset[i]即为一个user的train set and test set
    #     return cold_user_data #train set and test set of cold users based on leave-one-out

    # 选择负样本，直接拼接到训练集末尾
    def gen_neg_train(self, supx_set, qurx_set, neg_number=None):
        if neg_number is None:
            print("dont't select neg sample numbers!!")
            return None
        item_ids = self.item_data.movie_id.unique()
        random.shuffle(item_ids)
        all_data = pd.concat([supx_set, qurx_set], axis=0)
        inter_items = all_data.movie_id.unique()

        neg_inter = pd.DataFrame(columns=['user_id', "movie_id"])
        neg_inter_y = pd.DataFrame(columns=['rating'])
        for item in item_ids:
            if item not in inter_items:
                neg_inter = neg_inter.append(
                    pd.DataFrame([[supx_set.iloc[-1].user_id, item]], columns=['user_id', 'movie_id']))
                # neg_inter = neg_inter.append(neg_inter)
                neg_inter_y = neg_inter_y.append(pd.DataFrame([0], columns=['rating']))
                neg_number -= 1
                if neg_number <= 0: break
        return neg_inter.reset_index(drop=True), neg_inter_y.reset_index(drop=True)

    # add new users to history dataset after each roung of training
    def add_new_user_to_history(self, stage_id, types="not_only_new"):
        if types == "not_only_new":
            history_path = self.path + "/stage_dataset/{}".format(stage_id)
        else:
            history_path = self.path + "/stage_dataset_only_new/{}".format(stage_id)
        cold_user_path = self.path + "/cold_user/{}".format(stage_id)
        max_idx = len(os.listdir(history_path)) // 4

        for idx in range(len(os.listdir(cold_user_path)) // 4):
            train_x = pd.read_csv(cold_user_path + '/supp_x_{}.dat'.format(idx), names=['user_id', 'movie_id'],
                                  engine='python')
            train_y = pd.read_csv(cold_user_path + '/supp_y_{}.dat'.format(idx), names=['rating'], engine='python')
            test_x = pd.read_csv(cold_user_path + '/query_x_{}.dat'.format(idx), names=['user_id', 'movie_id'],
                                 engine='python')
            test_y = pd.read_csv(cold_user_path + '/query_y_{}.dat'.format(idx), names=['rating'], engine='python')

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            train_x.to_csv("{}/supp_x_{}.dat".format(history_path, idx + max_idx), header=None,
                           columns=['user_id', 'movie_id'], index=False)
            train_y.to_csv("{}/supp_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'],
                           index=False)
            test_x.to_csv("{}/query_x_{}.dat".format(history_path, idx + max_idx), header=None,
                          columns=['user_id', 'movie_id'], index=False)
            test_y.to_csv("{}/query_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'],
                          index=False)
        # shutil.rmtree(cold_user_path)


class online_yelp(object):
    def __init__(self, stage_num=38):
        self.path = './datasets/yelp'
        # self.user_data, self.item_data, self.score_data = self.load()
        #
        #
        # # self.gen_attribution(self.item_data)
        # # self.score_data = self.filter_data(self.score_data)
        #
        # print(len(self.score_data), self.score_data.user_id.nunique(), self.score_data.item_id.nunique())
        # # user_ids = list(set(self.score_data.user_id))
        # self.user_num, self.item_num = self.user_data.user_id.nunique(), self.item_data.item_id.nunique()

        # self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num, self.score_data)
        # users=[]
        # items=[]
        # for i in tqdm(range(stage_num)):
        #     user,item=self.generate_current_suppAndQuery(i, types="only_new")
        #     users.append(user)
        #     items.append(item)
        #
        # self.user_data,self.item_data = self.getArise()

        self.u_dict, self.i_dict = self.generate(self.path)
        # warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num,warm_users_data)

        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()
        # # # # #generate each time(stage) support and query set
        # for i in tqdm(range(9,stage_num)):
        #     self.generate_current_suppAndQuery(i, types="only_new")

    # load all users and items and rating in row data
    def load(self):
        path = "./datasets/yelp/row"
        user_data_path = "{}/yelp.user".format(path)
        inter_data_path = "{}/yelp.inter".format(path)
        item_data_path = "{}/yelp.item".format(path)
        # user_id:token	user_review_count:float	yelping_since:float	user_useful:float	user_funny:float	user_cool:float	fans:float	average_stars:float	compliment_hot:float	compliment_more:float	compliment_profile:float	compliment_cute:float	compliment_list:float	compliment_note:float	compliment_plain:float	compliment_cool:float	compliment_funny:float	compliment_writer:float	compliment_photos:float

        user_data = pd.read_csv(user_data_path,
                                names=['user_id', 'user_name', 'user_review_count', 'yelping_since', 'user_useful',
                                       'user_funny', 'user_cool', 'elite', 'fans', 'average_stars', 'compliment_hot',
                                       'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list',
                                       'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny',
                                       'compliment_writer', 'compliment_photos'],
                                sep="\t", engine='python')
        user_data = user_data.drop(0)

        # 删除一些无关列
        # 可以根据user_review_count筛选用户
        user_data = user_data.drop(
            columns=['user_name', 'elite', 'yelping_since', 'user_useful', 'user_funny', 'user_cool', 'compliment_note',
                     'compliment_plain', 'compliment_cool', 'compliment_funny'])

        print("num of user attributions:", user_data.user_review_count.nunique(), user_data.fans.nunique(),
              user_data.average_stars.nunique(), user_data.compliment_hot.nunique(),
              user_data.compliment_more.nunique(), user_data.compliment_profile.nunique(),
              user_data.compliment_cute.nunique(), user_data.compliment_list.nunique(),
              user_data.compliment_writer.nunique(), user_data.compliment_photos.nunique())

        # city:token_seq	state:token	postal_code:token	latitude:float	longitude:float	item_stars:float	item_review_count:float	 is_open:float	categories:token_seq
        item_data = pd.read_csv(
            item_data_path,
            names=['item_id', 'item_name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude',
                   'item_stars', 'item_review_count', 'is_open', 'categories'],
            sep="\t", engine='python')
        item_data = item_data.drop(0)
        # item_data = item_data.dropna()
        item_data = item_data.drop(columns=['item_name', 'address', 'latitude', 'longitude'])

        item_data['city'].fillna(method='bfill', inplace=True)
        item_data['postal_code'].fillna(method='bfill', inplace=True)
        print("item attribution num:", item_data.city.nunique(), item_data.state.nunique(),
              item_data.postal_code.nunique(), item_data.item_stars.nunique(), item_data.item_review_count.nunique())

        # review_id:token	user_id:token	business_id:token	stars:float	useful:float	funny:float	cool:float	date:float
        score_data = pd.read_csv(
            inter_data_path, names=['review_id', 'user_id', 'item_id', 'stars', 'useful', 'funny', 'cool', 'timestamp'],
            sep="\t", engine='python')
        score_data = score_data.drop(0)
        score_data = score_data.drop(columns=['review_id', 'useful', 'funny', 'cool'])
        score_data = score_data.sort_values(by=['timestamp'])  # 按时间戳升序
        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(int(x)))
        score_data = score_data.drop(["timestamp"], axis=1)

        return user_data, item_data, score_data

    def filter_data(self, score):
        all_user = score.user_id.unique()

        filter_score = pd.DataFrame(columns=['user_id', 'item_id', 'stars', 'useful', 'funny', 'cool', 'timestamp'])
        for user in all_user:
            if len(score[score['user_id'] == user]) > 10:
                # suff_users.append(user)
                # if len(score[score['user_id']==user])<10:
                #     score.drop(score.index[(score['user_id']==user)],inplace=True)

                # u_info = score.loc[score.user_id == user].reset_index(drop=True)
                filter_score = pd.concat([filter_score, score.loc[score.user_id == user]])
        return filter_score

    def gen_attribution(self, item_data):  # state:token	postal_code:token
        file_path = self.path + "/row/"
        cate_list = item_data.categories.unique()
        categories_list = []
        for c in cate_list:
            categories_list += [re.sub(r'\'', '', cat).strip().strip() for cat in set(c.split(
                ', '))]  # 去掉左边和右边的'   [re.sub(r'\'', '', cat).strip() for cat in set(c.split(','))]
        categories_list = list(set(categories_list))

        city_list = item_data.city.unique()
        state_list = item_data.state.unique()
        postal_code_list = item_data.postal_code.unique()
        self.write_attr(file_path + 'state.txt', state_list)
        self.write_attr(file_path + 'postal.txt', postal_code_list)
        self.write_attr(file_path + 'city.txt', city_list)
        self.write_attr(file_path + 'cate.txt', categories_list)

        print("num of categories", len(categories_list))

        # 其余都是float取值类型

    def write_attr(self, fname, data):
        f = open(fname, 'w', encoding="utf-8")
        for value in data:
            f.write(str(value) + '\n')

    # split score data by timestamp as online data
    '''
    :param load_dataset:movielens_1m.load()
    train and test:  80:20
    leave-one-out evaluation 
    写入数据集
    先划分stage_num个数据片，然后划分每个数据片（8:2）为train and test

    划分用户的interaction为train-set and test-set
    warm-user and cold-user
    '''

    def split_warm_cold(self, rat_data, max_count=10):
        storing_path = './datasets'
        dataset = 'amazon'
        # 按照时间划分数据集
        sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        print('start time %s,  end time %s' % (start_time, end_time))
        user_warm_list, user_cold_list, user_counts = [], [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm, cold = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'time']), pd.DataFrame(
            columns=['user_id', 'item_id', 'rating', 'time'])
        for u_id in user_ids:
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的

            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id] < 6 or u_count < 6: continue
            if self.user_current_interNum[u_id] <= 25:
                user_cold_list.append(u_id)
                self.cold_users.append(u_id)
                self.cold_users_inter = pd.concat([self.cold_users_inter, u_info]).reset_index(drop=True)

            else:
                user_warm_list.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
                if u_id in self.cold_users:
                    self.cold_users.remove(u_id)
                    warm = pd.concat(
                        [warm, self.cold_users_inter[self.cold_users_inter['user_id'] == u_id]]).reset_index(
                        drop=True)
                    self.cold_users_inter.drop(
                        self.cold_users_inter[self.cold_users_inter['user_id'] == u_id].index)
            user_counts.append(u_count)
        # cold = pd.concat([cold, u_info]).reset_index(drop=True)  # 取出目前为止该用户的所有交互

        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('min count: %d, sum count: %d, max count: %d' % (
            min(user_counts), sum(user_counts), max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        # user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
        #                   'user_cold_ids': user_cold_list}
        # pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
        return user_warm_list, user_cold_list, warm, self.cold_users_inter  # pd.DataFrame(warm,columns=['user_id', 'movie_id', 'rating', 'time']),pd.DataFrame(cold,columns=['user_id', 'movie_id', 'rating', 'time'],)

    # 对warm_user的数据生成streaming data
    def generate_streaming(self, stage_num, rate_d):
        sorted_time = rate_d.sort_values(by='time', ascending=True).reset_index(drop=True)
        stage_data_num = len(rate_d) // stage_num
        for i in tqdm(range(stage_num)):
            stage_data = sorted_time.iloc[i * stage_data_num:(i + 1) * stage_data_num, :]
            stage_data = pd.DataFrame(stage_data)
            stage_data.to_csv(self.path + '/streaming/' + str(i) + '.dat', index=False)

    '''
    based on choice of types to determine only current streaming or current and previous streaming to load dataset
    return current_all_users,current_all_inter(u_id and m_id),current_all_inter_rating(rate of u_id and m_id)
    '''

    def get_next_dataset(self, stage_id, types="only_new"):
        current_all_users, current_all_items, current_all_inter = [], set(), []
        try:
            # cold_users={}#leave-one-out:选择给定时间之后交互较少的用户作为cold-user，用于test-on-cold-users
            # current_stage_users=[]
            if types == "not_only_new":  # 将之前所有streaming的数据整合到一起作为历史数据
                score_data = pd.read_csv(self.path + '/streaming/' + str(stage_id) + '.dat',
                                         names=['user_id', 'item_id', 'rating', 'time'], engine='python')
                for i in range(0, stage_id):  # 将当前stage之前的数据
                    score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                    s_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating', 'time'],
                                         engine='python')
                    s_data = s_data.drop(0)
                    score_data = pd.concat([score_data, s_data])
                    score_data.reset_index(drop=True)
                warm_users, warm_users_data = self.split_warm_cold2(score_data)
                return warm_users, warm_users_data
            else:
                score_data_path = self.path + '/streaming/' + str(stage_id) + '.dat'
                score_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating', 'time'],
                                         engine='python')
                score_data = score_data.drop(0)
                warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(
                    score_data)  # cold and warm under current streaming

                # 选择在给定时间之后交互很少的用户作为cold-users（选择在当前streaming中交互很少的用户作为冷启动用户）
                return warm_users, cold_users, warm_users_data, cold_users_data
        except:
            print("read stage data wrong , may be there is no new data,finished")
            return None, None, None, None

    '''消融实验。获得full tune数据集，当下'''

    def get_current_full(self, stage_id):
        try:
            score_data = pd.read_csv(self.path + '/streaming/' + str(stage_id) + '.dat',
                                     names=['user_id', 'item_id', 'rating', 'time'],
                                     engine='python')
            users = score_data.user_id.unique()[1:]
            for i in range(0, stage_id):  # 将当前stage之前的数据
                score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                s_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating', 'time'],
                                     engine='python')
                score_data = pd.concat([score_data, s_data[s_data['user_id'].isin(users)]])
                score_data.reset_index(drop=True)
            warm_users, warm_users_data = self.split_warm_cold2(score_data)
            return warm_users, warm_users_data
        except:
            print("read stage data wrong , may be there is no new data,finished")
            return None, None, None, None

    def split_warm_cold2(self, rat_data, max_count=10):
        storing_path = '../datasets'
        dataset = 'movielens'
        # 按照时间划分数据集
        sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        # print('start time %s,  end time %s' % (start_time, end_time))
        warm_users, user_counts = [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'time'])
        for u_id in user_ids:
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)

            if u_count < 8: continue  # ************过滤数据太少的用户

            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if u_id not in self.cold_users:
                warm_users.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
            user_counts.append(u_count)
        print('min count: %d, sum count: %d, max count: %d' % (
            min(user_counts), sum(user_counts), max(user_counts)))
        return warm_users, warm

    '''get all users and interaction and rating y based on interaction dataset score_data'''

    def get_inter_y(self, score_data):
        inters = []
        for idx, row in score_data.iterrows():
            if row['user_id'] == 'user_id':
                continue
            # if row['user_id'] not in users:
            #     users.append(row['user_id'])
            # current_all_items.add(row['movie_id'])
            inters.append([row['user_id'], row['item_id'], float(row['rating']), row['time']])

        users_movie_inter = collections.defaultdict(list)
        users_movie_inter_y = collections.defaultdict(list)
        for row in inters:  # 因为stage_data中的数据是按时间排序的
            # if row[-2]=='rating':continue#跳过表头
            users_movie_inter[row[0]].append(row[1])  # 直接取前80%就是时间上前80%的交互
            users_movie_inter_y[row[0]].append(row[-2])
        return users_movie_inter, users_movie_inter_y

    '''分场景，将每个streaming data中的users划分为support-set and query-set
    先对每个streaming data中的用户看过的电影，以及评分y

    随机选择前80%的交互作为supprot set，其余20%交互为query-set
    '''

    # not only new
    # test and val:choose the latest items as the test and validation instance following the leave-one-out evaluation
    # 根据时间预先划分streaming data
    def generate_current_suppAndQuery(self, stage_id, types="only_new"):
        # 基于get_next的数据集采样用户，生成这些用户的support-set and query-set
        # current_all_users, users_movie_inter, users_movie_inter_y = self.get_next_dataset(stage_id)
        warm_users, cold_users, warm_users_data, cold_users_data = self.get_next_dataset(stage_id, types="only_new")

        return warm_users + cold_users, list(
            set(list(warm_users_data.item_id.unique()) + list(cold_users_data.item_id.unique())))

        if not warm_users:
            print("***********read stage data wrong , may be there is no new data,finished***************")
            return None
        self.generate_warm_sq(stage_id, warm_users, warm_users_data, types="only_new")

        if cold_users:
            cold_inter, cold_inter_y = self.get_inter_y(cold_users_data)
            self.generate_cold_dataset(cold_users, cold_inter, cold_inter_y, stage_id)
        else:
            print("************this {} streaming has not cold users!***********".format(stage_id))
        # warm_users_full, warm_users_data_full = self.get_current_full(stage_id)
        # self.generate_warm_sq(stage_id, warm_users_full, warm_users_data_full, types="only_new_full")

        self.user_current_interNum = {}
        # warm_users_f, warm_users_data_f = self.get_next_dataset(stage_id, types="not_only_new")
        # self.generate_warm_sq(stage_id, warm_users_f, warm_users_data_f, types="not_only_new")
        print("**********generate dataset successfully******************")
        return True
        # 是写入文件还是直接返回suppor and query

    def generate_warm_sq(self, stage_id, warm_users, warm_users_data, types="only_new"):
        warm_inter, warm_inter_y = self.get_inter_y(warm_users_data)

        if types == "not_only_new":
            dir = "stage_dataset"
        elif types == "only_new_full":
            dir = "stage_full_tunning"
        else:
            dir = "stage_dataset_only_new"

        users = warm_users
        n = len(users)
        # generate the support set and query set
        idx = 0
        for u_id in users:  # 有时间顺序的
            support_x = []
            query_x = []
            seen_movie = len(warm_inter[u_id])
            indices = list(range(seen_movie))
            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set

            tmp_x = np.array(warm_inter[u_id])
            tmp_y = np.array(warm_inter_y[u_id])

            if seen_movie <= 5:
                s_user = pd.DataFrame([u_id] * floor(seen_movie * 0.8), columns=['user_id'])
                q_user = pd.DataFrame([u_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['item_id'])
                query_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['item_id'])
                support_x = s_user.join(support_x)
                query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])
            else:
                s_user = pd.DataFrame([u_id] * (seen_movie - 5), columns=['user_id'])
                q_user = pd.DataFrame([u_id] * 5, columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:-5]], columns=['item_id'])
                query_x = pd.DataFrame(tmp_x[indices[-5:]], columns=['item_id'])
                support_x = s_user.join(support_x)
                query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:-5]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[-5:]], columns=['rating'])

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            folder_path = "{}/{}/{}".format(self.path, dir, stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            support_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['user_id', 'item_id'], index=False)
            support_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['rating'], index=False)
            query_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['user_id', 'item_id'], index=False)
            query_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['rating'], index=False)
            idx += 1

    def get_cold_inter_y(self, cold_users_data):
        cu_movie_inter = collections.defaultdict(list)
        cu_movie_inter_y = collections.defaultdict(list)
        # each_stage_num=len(cold_user_idx)//stage_num
        for idx, row in cold_users_data.iterrows():  # 因为stage_data中的数据是按时间排序的
            cu_movie_inter[row['user_id']].append(row['item_id'])  # 直接取前80%就是时间上前80%的交互
            cu_movie_inter_y[row['user_id']].append(row['rating'])
        return cu_movie_inter, cu_movie_inter_y

    # 直接对每个cold
    def generate_cold_dataset(self, cold_user_idx, cu_movie_inter, cu_movie_inter_y, stage_id):
        idx = 0
        for cold_id in cold_user_idx:

            seen_movie = len(cu_movie_inter[cold_id])
            indices = list(range(seen_movie))

            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set
            tmp_x = np.array(cu_movie_inter[cold_id])
            tmp_y = np.array(cu_movie_inter_y[cold_id])
            train_user = pd.DataFrame([cold_id] * floor(seen_movie * 0.8), columns=['user_id'])
            test_user = pd.DataFrame([cold_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

            train_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['item_id'])
            train_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
            train_x = train_user.join(train_x)

            test_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['item_id'])
            test_x = test_user.join(test_x)
            test_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            folder_path = "{}/{}/{}".format(self.path, 'cold_user', stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            train_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['user_id', 'item_id'], index=False)
            train_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['rating'], index=False)
            test_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['user_id', 'item_id'], index=False)
            test_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['rating'], index=False)
            idx += 1

    '''['city', 'state', 'postal_code',  'item_stars','item_review_count', 'is_open', 'categories']

           ['user_review_count','fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
            'compliment_list', 
            'compliment_writer', 'compliment_photos']'''

    # 10
    def user_converting(self, row, u_count_list, fans_list, average_stars_list, c_hot_list, c_more_list, c_profile_list,
                        c_cute_list, c_list_list, c_writer_list, c_photos_list):
        # gender_dim: 2, age_dim: 7, occupation: 21
        count_idx = u_count_list.index(str(row['user_review_count']))
        fans_idx = fans_list.index(str(row['fans']))
        average_stars_idx = average_stars_list.index(str(row['average_stars']))
        c_hot_idx = c_hot_list.index(str(row['compliment_hot']))

        c_more_idx = c_more_list.index(str(row['compliment_more']))
        c_profile_idx = c_profile_list.index(str(row['compliment_profile']))
        c_cute_idx = c_cute_list.index(str(row['compliment_cute']))
        c_list_idx = c_list_list.index(str(row['compliment_list']))
        c_writer_idx = c_writer_list.index(str(row['compliment_writer']))
        c_photos_idx = c_photos_list.index(str(row['compliment_photos']))

        return [count_idx, fans_idx, average_stars_idx, c_hot_idx, c_more_idx, c_profile_idx, c_cute_idx, c_list_idx,
                c_writer_idx, c_photos_idx]

    # 6
    def item_converting(self, row, city_list, state_list, post_list, cate_list, i_stars_list,
                        i_count_list):  # ,is_open_list):
        # rate_dim: 6, year_dim: 1,  genre_dim:25, director_dim: 2186,
        city_idx = city_list.index(str(row['city']).strip())
        state_idx = state_list.index(str(row['state']).strip())
        post_idx = post_list.index(str(row['postal_code']).strip())
        i_star_idx = i_stars_list.index(str(row['item_stars']).strip())
        count_idx = i_count_list.index(str(row['item_review_count']).strip())
        # open_idx = is_open_list.index(str(row['is_open']))

        cate_idx = [0] * 1377
        # for cate in str(row['genre']).split(", "):
        #     idx = genre_list.index(genre)
        #     genre_idx[idx] = 1
        # director_idx = [0] * 2186
        for cate in str(row['categories']).split(", "):
            idx = cate_list.index(re.sub(r'\'', '', cate))  # re.sub(r'\([^()]*\)', '', cate)
            cate_idx[idx] = 1
        out_list = list([city_idx, state_idx, post_idx, i_star_idx, count_idx])  # ,open_idx
        out_list.extend(cate_idx)
        return out_list

    def load_list(self, fname):
        list_ = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():
                list_.append(line.strip())
        return list_

    '''dataset = movielens_1m()
    item_data=dataset.item_data
    user_data=dataset.user_data
    '''
    '''['city', 'state', 'postal_code',  'item_stars','item_review_count','categories']

        ['user_review_count','fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
         'compliment_list', 'compliment_writer', 'compliment_photos']'''

    def getArise(self, users, items):
        users_p = pd.DataFrame(
            columns=['user_review_count', 'fans', 'average_stars', 'compliment_hot', 'compliment_more',
                     'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_writer',
                     'compliment_photos'])
        items_p = pd.DataFrame(
            columns=['city', 'state', 'postal_code', 'item_stars', 'item_review_count', 'categories'])
        for u in users:
            u_info = self.user_data.loc[self.user_data.user_id == u]
            users_p = pd.concat([users_p, u_info])

        for i in items:
            i_info = self.item_data.loc[self.item_data.item_id == i]
            items_p = pd.concat([items_p, i_info])

        users_p.reset_index(drop=True)
        items_p.reset_index(drop=True)
        return users_p, items_p

    def generate(self, master_path):

        # hashmap for item information
        if not os.path.exists("{}/item_dict.pkl".format(master_path)):
            # item
            city_list = self.load_list("{}/{}/city.txt".format(self.path, 'row'))
            state_list = self.load_list("{}/{}/state.txt".format(self.path, 'row'))
            post_list = self.load_list("{}/{}/postal.txt".format(self.path, 'row'))
            cate_list = self.load_list("{}/{}/cate.txt".format(self.path, 'row'))
            i_stars_list = list(item_data.item_stars.unique())
            i_count_list = list(item_data.item_review_count.unique())
            is_open_list = list(item_data.is_open.unique())

            item_dict = {}
            for idx, row in item_data.iterrows():
                if row['item_id'] == 'item_id': continue  # 跳过表头
                i_info = self.item_converting(row, city_list, state_list, post_list, cate_list, i_stars_list,
                                              i_count_list)  # ,is_open_list)
                item_dict[row['item_id']] = i_info
            pickle.dump(item_dict, open("{}/item_dict.pkl".format(master_path), "wb"))
        else:
            item_dict = pickle.load(open("{}/item_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/user_dict.pkl".format(master_path)):
            # user
            u_count_list = list(user_data.user_review_count.unique())
            fans_list = list(user_data.fans.unique())
            average_stars_list = list(user_data.average_stars.unique())
            c_hot_list = list(user_data.compliment_hot.unique())
            c_more_list = list(user_data.compliment_more.unique())
            c_profile_list = list(user_data.compliment_profile.unique())
            c_cute_list = list(user_data.compliment_cute.unique())
            c_list_list = list(user_data.compliment_list.unique())
            c_writer_list = list(user_data.compliment_writer.unique())
            c_photos_list = list(user_data.compliment_photos.unique())
            user_dict = {}
            for idx, row in user_data.iterrows():
                if row['user_id'] == 'user_id': continue  # 跳过表头
                u_info = self.user_converting(row, u_count_list, fans_list, average_stars_list, c_hot_list, c_more_list,
                                              c_profile_list, c_cute_list, c_list_list, c_writer_list, c_photos_list)
                user_dict[row['user_id']] = u_info
            pickle.dump(user_dict, open("{}/user_dict.pkl".format(master_path), "wb"))
        else:
            user_dict = pickle.load(open("{}/user_dict.pkl".format(master_path), "rb"))
        return user_dict, item_dict

    '''打开下一时刻的streaming data
    movielens/support/supp_x_stageid_{}.pkl    movielens/support/supp_y_stageid_{}.pkl
    movielens/query/query_x_stageid_{}.pkl     movielens/query/query_y_stageid_{}.pkl 
    x_stageid_{}.pkl  y_stageid_{}.pkl
    D_test=D_val
    一个stage_id对应一个streaming
    return: total_dataset
    根据dataset_path来生成warm and cold的数据集
    warm:datasets/movielens/stage_dataset        cold:datasets/movielens/cold_user
    '''

    def next_sampling_tasks(self, sample_num, stage_id, dataset_path, types="not_only_new", neg_num=None):
        if types == "not_only_new":
            dir = "/stage_dataset/"
        elif types == "test_cold":
            dir = "/cold_user/"
        else:
            dir = "/stage_dataset_only_new/"
        dataset_path = dataset_path + dir + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
            return None
        all_tasks = [idx for idx in range(len(data_file) // 4)]
        random.shuffle(all_tasks)
        sampled_tasks = all_tasks[:sample_num]
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        neg_x_s = []
        neg_y_s = []
        users_id = []
        for idx in sampled_tasks:
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            u_id = None
            if len(s_x["user_id"]) < 1: continue  # 过滤空文件
            inter_item = 0
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                u_id = int(row['user_id'])
                inter_item += 1
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[str(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    a = self.u_dict[row['user_id']]
                    support_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.m_dict[row['item_id']]])
                # s_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
                # except:
                #     support_x= s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted
            if u_id:
                users_id.append(u_id)

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            for idx, row in q_x.iterrows():
                if row['item_id'] == "item_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[str(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[str(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.m_dict[row['item_id']]])

                # q_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     query_x= torch.cat((query_x, q_x_converted), 0)#按维数0（行）拼接
                # except:
                #     query_x= q_x_converted
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

            if neg_num:
                neg_inter, neg_inter_y = self.gen_neg_train(s_x, q_x,
                                                            neg_number=inter_item)  # the number of neg_item is equal to pos_item number
                neg_x = [None]
                neg_y = [None]
                for idx, row in neg_inter.iterrows():
                    if row['item_id'] == "item_id": continue
                    # try:
                    #     neg_x[0] = torch.cat((neg_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])), 0)  # 按维数0（行）拼接
                    # except:
                    #     neg_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                    # try:
                    #     neg_x[1] = torch.cat((neg_x[1], torch.tensor([self.m_dict[row['movie_id']]])),
                    #                              0)  # 按维数0（行）拼接
                    # except:
                    #     neg_x[1] = torch.tensor([self.m_dict[row['movie_id']]])

                    try:
                        neg_x = torch.cat((neg_x, torch.tensor([self.m_dict[row['item_id']]])), 0)  # 按维数0（行）拼接
                    except:
                        neg_x = torch.tensor([self.m_dict[row['item_id']]])
                    s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
                    try:
                        neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
                    except:
                        neg_y = s_y_converted
                neg_x_s.append(neg_x)
                neg_y_s.append(neg_y)

        if neg_num:
            d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
        else:
            d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t, users_id

    def next_dataset(self, stage_id, dataset_path, types="only_new"):
        if types == "not_only_new":
            dir = "/stage_dataset/"
        elif types == "test_cold":
            dir = "/cold_user/"
        else:
            dir = "/stage_dataset_only_new/"
        dataset_path = dataset_path + dir + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        u_ids = set()
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            u_id = None
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['item_id'] == "item_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                u_id = str(row['user_id'])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[str(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[str(row['user_id'])]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.i_dict[row['item_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.i_dict[row['item_id']]])
                # s_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
                # except:
                #     support_x= s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            if u_id: u_ids.add(u_id)
            for idx, row in q_x.iterrows():
                if row['item_id'] == "item_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[str(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[str(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.i_dict[row['item_id']]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.i_dict[row['item_id']]])

                # q_x_converted = torch.cat((self.m_dict[int(row['movie_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     query_x= torch.cat((query_x, q_x_converted), 0)#按维数0（行）拼接
                # except:
                #     query_x= q_x_converted
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

        d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t, list(u_ids)

    # add new users to history dataset after each roung of training
    def add_new_user_to_history(self, stage_id, types="only_new"):
        if types == "not_only_new":
            history_path = self.path + "/stage_dataset/{}".format(stage_id)
        else:
            history_path = self.path + "/stage_dataset_only_new/{}".format(stage_id)
        cold_user_path = self.path + "/cold_user/{}".format(stage_id)
        max_idx = len(os.listdir(history_path)) // 4

        for idx in range(len(os.listdir(cold_user_path)) // 4):
            train_x = pd.read_csv(cold_user_path + '/supp_x_{}.dat'.format(idx), names=['user_id', 'item_id'],
                                  engine='python')
            train_y = pd.read_csv(cold_user_path + '/supp_y_{}.dat'.format(idx), names=['rating'], engine='python')
            test_x = pd.read_csv(cold_user_path + '/query_x_{}.dat'.format(idx), names=['user_id', 'item_id'],
                                 engine='python')
            test_y = pd.read_csv(cold_user_path + '/query_y_{}.dat'.format(idx), names=['rating'], engine='python')

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            train_x.to_csv("{}/supp_x_{}.dat".format(history_path, idx + max_idx), header=None,
                           columns=['user_id', 'item_id'], index=False)
            train_y.to_csv("{}/supp_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'],
                           index=False)
            test_x.to_csv("{}/query_x_{}.dat".format(history_path, idx + max_idx), header=None,
                          columns=['user_id', 'item_id'], index=False)
            test_y.to_csv("{}/query_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'],
                          index=False)
        # shutil.rmtree(cold_user_path)

    def next_full_dataset(self, stage_id, dataset_path):
        path = "/stage_full_tunning/"
        dataset_path = dataset_path + path + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
            return
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['item_id'] == "item_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.m_dict[row['item_id']]])
                # s_x_converted = torch.cat((self.m_dict[int(row['item_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
                # except:
                #     support_x= s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            for idx, row in q_x.iterrows():
                if row['item_id'] == "item_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.m_dict[row['item_id']]])
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

        d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t

    def next_full_data(self, stage_id, path):
        # 读取<=stage_id之前的所有stage_dataset_only_new里的数据，然后如果support_x里面的user相同的话就合并
        users = collections.OrderedDict()
        for stage in range(stage_id + 1):
            dataset_path = path + "/stage_dataset_only_new/" + str(stage)
            try:  # get all filename
                data_file = os.listdir(dataset_path)
            except:
                print("filename error or filepath error!")
            supp_x_s, supp_y_s = [], []
            query_x_s, query_y_s = [], []
            for idx in range(len(data_file) // 4):
                s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                                  header=None,
                                  engine='python')
                s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], header=None,
                                  engine='python')
                q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                                  header=None,
                                  engine='python')
                q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], header=None,
                                  engine='python')
                support_x = [None, None]
                query_x = [None, None]
                support_y = None
                query_y = None
                if len(s_x["user_id"]) < 1: continue
                for idx, row in s_x.iterrows():
                    if row['item_id'] == "item_id": continue
                    try:
                        support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[row['user_id']]])),
                                                 0)  # 按维数0（行）拼接
                    except:
                        support_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                    try:
                        support_x[1] = torch.cat((support_x[1], torch.tensor([self.i_dict[row['item_id']]])),
                                                 0)  # 按维数0（行）拼接
                    except:
                        support_x[1] = torch.tensor([self.i_dict[row['item_id']]])

                    s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                    try:
                        support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                    except:
                        support_y = s_y_converted
                for idx, row in q_x.iterrows():
                    if row['item_id'] == "item_id": continue
                    # query_x[0].append(self.u_dict[int(row['user_id'])])
                    # query_x[1].append(self.m_dict[int(row['movie_id'])])
                    try:
                        query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[row['user_id']]])),
                                               0)  # 按维数0（行）拼接
                    except:
                        query_x[0] = torch.tensor([self.u_dict[row['user_id']]])
                    try:
                        query_x[1] = torch.cat((query_x[1], torch.tensor([self.i_dict[row['item_id']]])),
                                               0)  # 按维数0（行）拼接
                    except:
                        query_x[1] = torch.tensor([self.i_dict[row['item_id']]])
                    q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                    try:
                        query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                    except:
                        query_y = q_y_converted

                user = s_x.user_id.unique()[0]
                if s_x.user_id.unique()[0] in users.keys():
                    # filter_score = pd.concat([filter_score, score.loc[score.user_id == user]])
                    # users[s_x.user_id.unique()][0] = pd.concat([users[s_x.user_id.unique()][0], s_x])
                    # users[s_x.user_id.unique()][1] = pd.concat([users[s_x.user_id.unique()][1], s_y])
                    # users[s_x.user_id.unique()][2] = pd.concat([users[s_x.user_id.unique()][2], q_x])
                    # users[s_x.user_id.unique()][3] = pd.concat([users[s_x.user_id.unique()][3], q_y])
                    users[user][0][0] = torch.cat((users[user][0][0], support_x[0]), 0)
                    users[user][0][1] = torch.cat((users[user][0][1], support_x[1]), 0)
                    users[user][1] = torch.cat((users[user][1], support_y), 0)
                    users[user][2][0] = torch.cat((users[user][2][0], query_x[0]), 0)
                    users[user][2][1] = torch.cat((users[user][2][1], query_x[1]), 0)
                    users[user][3] = torch.cat((users[user][3], query_y), 0)
                else:
                    users[user] = [support_x, support_y, query_x, query_y]
        return list(users.values()), binary_conversion(sys.getsizeof(list(users.values())))

    def sample_tasks(self, data, sample_num):
        tasks = list(data.keys())
        random.shuffle(tasks)

        dataset = []
        for t in tasks[:sample_num]:
            dataset.append(data[t])

        return dataset

    def get_all_user_item(self, stage_id, path):
        user_dict = pickle.load(open("{}/user_dict.pkl".format(path), "rb"))
        appear_user = {}
        items = set()
        interactions = 0
        for stage in range(stage_id + 1):
            dataset_path = path + "/stage_dataset_only_new/" + str(stage)
            try:  # get all filename
                data_file = os.listdir(dataset_path)
            except:
                print("filename error or filepath error!")

            for idx in range(len(data_file) // 4):

                s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                                  header=None,
                                  engine='python')
                q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                                  header=None,
                                  engine='python')
                interactions += len(s_x)
                interactions += len(q_x)
                user = s_x.user_id.unique()[0]
                appear_user[user] = user_dict[user]
                for idx, row in s_x.iterrows():
                    items.add(row['item_id'])

                for idx, row in q_x.iterrows():
                    items.add(row['item_id'])

            c_dataset_path = path + "/cold_user/" + str(stage)
            try:  # get all filename
                c_data_file = os.listdir(c_dataset_path)
            except:
                print("filename error or filepath error!")

            for idx in range(len(c_data_file) // 4):
                s_x = pd.read_csv('{}/supp_x_{}.dat'.format(c_dataset_path, idx), names=['user_id', 'item_id'],
                                  header=None,
                                  engine='python')
                q_x = pd.read_csv('{}/query_x_{}.dat'.format(c_dataset_path, idx), names=['user_id', 'item_id'],
                                  header=None,
                                  engine='python')
                interactions += len(s_x)
                interactions += len(q_x)
                user = s_x.user_id.unique()[0]
                appear_user[user] = user_dict[user]
                for idx, row in s_x.iterrows():
                    items.add(row['item_id'])

                for idx, row in q_x.iterrows():
                    items.add(row['item_id'])
        print(interactions, len(items))
        # pickle.dump(appear_user, open("{}/user_dict2.pkl".format(path), "wb"))


class trainDataset_withPreSample(Dataset):
    '''
    this is Dataset type for train transfer, the input dataset has sampled enough
    neg item. the each epoch, will select on cloumn neg_item as neg item
    '''

    def __init__(self, input_dataset):
        super(trainDataset_withPreSample, self).__init__()
        self.all_data = copy.deepcopy(input_dataset)
        self.have_read = 0
        self.neg_flag = np.arange(1, self.all_data.shape[1])
        np.random.shuffle(self.neg_flag)
        self.neg_all = input_dataset.shape[1] - 2
        self.used_neg_count = 0
        self.data_len = input_dataset.shape[0]

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        user = self.all_data[idx, 0]
        item = self.all_data[idx, 1]
        neg_item = self.all_data[idx, self.neg_flag[self.used_neg_count]]
        self.have_read += 1
        if self.have_read >= self.data_len:
            self.have_read = 0
            self.used_neg_count += 1
            if self.used_neg_count >= self.neg_all:
                np.random.shuffle(self.neg_flag)
                self.used_neg_count = 0
        return user, item, neg_item


class online_data():
    def __init__(self, args, path="dataset/", datasetname="News", online_train_time=21, file_path_list=None,
                 test_list=None, validation_list=None, online_test_time=48):
        '''
        :param args:  parameters collection
        :param path:  data path
        :param datasetname: dataset name
        :param online_train_time: the stage start to training
        :param file_path_list: file path list :such as , for yelp-- [0,1, ..., 39] i.e stage ids
        :param test_list:  test path list : such as for yelp-- [30,31,...,39]
        :param validation_list: not used!!
        :param online_test_time: the stage id that start to online testing
        '''
        self.TR_sample_type = args.TR_sample_type  # TR sample typr
        self.TR_stop_ = args.TR_stop_  # if stop train transfer when online test stages
        self.MF_sample = args.MF_sample
        self.current_as_set_tt = args.set_t_as_tt
        self.path = path
        self.dataname = datasetname
        self.file_list = file_path_list
        self.test_list = test_list
        self.val_list = validation_list
        self.len = len(file_path_list)
        self.online_trian_time = online_train_time
        self.online_test_time = online_test_time
        self.start_test_time = online_test_time  # from which time, online train
        self.test_count = 0  # start as test
        information = np.load(self.path + self.dataname + "/" + "information.npy")
        self.user_number = information[1]
        self.inter_all = information[0]
        self.item_number = information[2]
        print(information)
        a = [np.load(self.path + self.dataname + "/train/" + file + ".npy") for file in self.file_list]
        a = np.concatenate(a, axis=0)
        print(a.shape[0], np.unique(a[:, 0]).shape[0], np.unique(a[:, 1]).shape[0])
        del a

    def reinit(self):  # used for args.pass_num great than 1, reint the dataset
        self.test_count = 0
        self.start_test_time = copy.deepcopy(self.online_test_time)

    def next_train(self, d_time):
        """
        d_time: the diss tance between self.online_trian_time and now_time
        in online train stage we need D_t and D_(t+1) ,different from D_(t+1) in test, we use it to
        train rec save_model,so loss should be compute.
        :return:
         set:      D_t
         set_tt:   D_(t+1)
         stop_: if ture ,stop train when online train stage
        """
        now_time = self.online_trian_time + d_time
        if (now_time + 1) >= self.len:  # end!!
            return None, None, None, None
        print("now time:", now_time)
        print("will be test data:", now_time + 1)
        if (now_time + 1) < self.start_test_time:  # stop online train，if stop, we will stop online train
            if self.MF_sample == "alone":
                set_t = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
            elif self.MF_sample == 'all':
                set_t = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
            else:
                raise TypeError("now such type when read next train sets")

            val = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")

            print("now time:", self.file_list[now_time])
            if self.TR_sample_type == 'alone':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                    print("set tt is:", self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time + 1] + ".npy")
                    print("set_tt is:", self.path + self.dataname + "/train/" + self.file_list[now_time + 1] + ".npy")
            elif self.TR_sample_type == 'all':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
                print("t+1 data stes", self.file_list[now_time + 1])
            else:
                raise TypeError("no such TR sample type")

            return set_t, set_tt, None, val
        elif self.TR_stop_:  # stop train transfer when online test satge
            if self.MF_sample == "alone":
                set_t = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
            elif self.MF_sample == 'all':
                set_t = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
            else:
                raise TypeError("now such type when read next train sets")

            print("now time:", self.file_list[now_time])
            print("will be test data:", self.test_list[self.test_count])
            now_test = np.load(self.path + self.dataname + "/test/" + self.test_list[self.test_count] + ".npy")
            val = now_test
            self.test_count += 1
            return set_t, None, now_test, val
        else:
            """
            if select this ,will still train in online stage,it should first do test, 
            then use the data to train transfer!! 
            """
            if self.MF_sample == "alone":
                set_t = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
            elif self.MF_sample == 'all':
                set_t = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
            else:
                raise TypeError("now such type when read next train sets")

            val = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
            '''according TR sample ty to load set_tt'''
            if self.TR_sample_type == 'alone':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                    print("settt is:", self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time + 1] + ".npy")
                    print("settt is:", self.path + self.dataname + "/train/" + self.file_list[now_time + 1] + ".npy")
                    # set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time + 1] + ".npy")
            elif self.TR_sample_type == 'all':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
                    print("set_tt is: ", self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
                    print("set_tt is", self.file_list[now_time + 1])
                # set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
                print("t+1 datasets,", self.file_list[now_time + 1])
            else:
                raise TypeError("no such TR sample type")

            now_test = np.load(self.path + self.dataname + "/test/" + self.test_list[self.test_count] + ".npy")
            print("real test:", self.test_list[self.test_count])
            self.test_count += 1
            return set_t, set_tt, now_test, val


'''
online Amazon dataset
split interactions into 50 periods.
'''


class online_amazon(object):
    def __init__(self, stage_num=60):
        self.path = './datasets/amazon'
        self.item_data, self.score_data = self.load()
        user_ids = list(set(self.score_data.user_id))
        self.user_num, self.item_num = self.score_data.user_id.nunique(), self.score_data.item_id.nunique()
        # self.gen_attribution(self.item_data)

        self.item_dict = self.generate(self.path)
        # warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num,warm_users_data)

        # self.generate_streaming(stage_num, self.score_data)
        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()
        # # # # #generate each time(stage) support and query set
        # for i in tqdm(range(stage_num)):
        #   self.generate_current_suppAndQuery(i, types="only_new")

    # load all users and items and rating in row data
    def load(self):
        path = "./datasets/amazon/row"
        inter_data_path = "{}/Amazon_Electronics.inter".format(path)
        item_data_path = "{}/Amazon_Electronics.item".format(path)

        # item_id:token	categories:token_seq	title:token	price:float	sales_type:token	sales_rank:float	brand:token
        item_data = pd.read_csv(
            item_data_path,
            names=['item_id', 'categories', 'title', 'price', 'sales_type', 'sales_rank', 'brand'],
            sep="\t", engine='python', encoding="utf-8"
        )
        item_data = item_data.drop(0)
        # item_data = item_data.dropna()
        # item_data = item_data.drop(columns=['released', 'writer', 'actors', 'plot', 'poster'])
        item_data = item_data.drop(columns=['sales_rank', 'sales_type'])
        # item_data=item_data[item_data['title'].notnull()]
        item_data['title'].fillna(method='bfill', inplace=True)
        price = item_data[item_data['price'].notnull()].price.tolist()
        avg_p = sum([float(i) for i in price]) / len(price)
        item_data['price'].fillna(str(avg_p), inplace=True)
        item_data['brand'].fillna(method='bfill', inplace=True)  # ffill:以前一个非NAN值填充

        # item_data['sales_type'].fillna('Electronics', inplace=True)

        # user_id:token	item_id:token	rating:float	timestamp:float
        score_data = pd.read_csv(
            inter_data_path, names=['user_id', 'item_id', 'rating', 'timestamp'],
            sep="\t", engine='python'
        )
        score_data = score_data.drop(0)
        score_data = score_data.sort_values(by=['timestamp'])  # 按时间戳升序

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(int(x)))
        score_data = score_data.drop(["timestamp"], axis=1)
        return item_data, score_data

    def gen_attribution(self, item_data):  # 'categories', 'title', 'price', 'sales_type', 'sales_rank', 'brand'
        file_path = self.path + "/row/"

        cate_list = item_data.categories.unique()
        categories_list = []
        for c in cate_list:
            categories_list += [eval(cat).strip() for cat in set(c.split(
                ','))]  # 去掉左边和右边的'   [re.sub(r'\'', '', cat).strip() for cat in set(c.split(','))]
        categories_list = list(set(categories_list))
        title_list = item_data.title.unique()
        price_list = item_data.price.unique()
        brand_list = item_data.brand.unique()
        print("number of categories {}, number of title {},number of price {},number of brand {}".format(
            len(categories_list), len(title_list), len(price_list), len(brand_list)))
        self.write_attr(file_path + 'categories.txt', categories_list)
        self.write_attr(file_path + 'title.txt', title_list)
        self.write_attr(file_path + 'price.txt', price_list)
        self.write_attr(file_path + 'brand.txt', brand_list)

    def write_attr(self, fname, data):
        f = open(fname, 'w', encoding="utf-8")
        for value in data:
            f.write(repr(value.strip()) + '\n')

    # split score data by timestamp as online data
    '''
    :param load_dataset:movielens_1m.load()
    train and test:  80:20
    leave-one-out evaluation 
    写入数据集
    先划分stage_num个数据片，然后划分每个数据片（8:2）为train and test

    划分用户的interaction为train-set and test-set
    warm-user and cold-user
    '''

    def split_warm_cold(self, rat_data, max_count=10):
        storing_path = './datasets'
        dataset = 'amazon'
        # 按照时间划分数据集
        sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        print('start time %s,  end time %s' % (start_time, end_time))
        user_warm_list, user_cold_list, user_counts = [], [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm, cold = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'time']), pd.DataFrame(
            columns=['user_id', 'item_id', 'rating', 'time'])
        for u_id in user_ids:
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的

            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id] < 10 or u_count < 2: continue
            if 5 < self.user_current_interNum[u_id] <= 30:
                user_cold_list.append(u_id)
                self.cold_users.append(u_id)
                self.cold_users_inter = pd.concat([self.cold_users_inter, u_info]).reset_index(drop=True)

            else:
                user_warm_list.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
                if u_id in self.cold_users:
                    self.cold_users.remove(u_id)
                    warm = pd.concat(
                        [warm, self.cold_users_inter[self.cold_users_inter['user_id'] == u_id]]).reset_index(
                        drop=True)
                    self.cold_users_inter.drop(
                        self.cold_users_inter[self.cold_users_inter['user_id'] == u_id].index)
            user_counts.append(u_count)
        # cold = pd.concat([cold, u_info]).reset_index(drop=True)  # 取出目前为止该用户的所有交互

        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('min count: %d, avg count: %d, max count: %d' % (
            min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        # user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
        #                   'user_cold_ids': user_cold_list}
        # pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
        return user_warm_list, user_cold_list, warm, self.cold_users_inter  # pd.DataFrame(warm,columns=['user_id', 'movie_id', 'rating', 'time']),pd.DataFrame(cold,columns=['user_id', 'movie_id', 'rating', 'time'],)

    # 对warm_user的数据生成streaming data
    def generate_streaming(self, stage_num, rate_d):
        sorted_time = rate_d.sort_values(by='time', ascending=True).reset_index(drop=True)
        stage_data_num = len(rate_d) // stage_num
        for i in tqdm(range(stage_num)):
            stage_data = sorted_time.iloc[i * stage_data_num:(i + 1) * stage_data_num, :]
            stage_data = pd.DataFrame(stage_data)
            stage_data.to_csv(self.path + '/streaming/' + str(i) + '.dat', index=False)

    '''
    based on choice of types to determine only current streaming or current and previous streaming to load dataset
    return current_all_users,current_all_inter(u_id and m_id),current_all_inter_rating(rate of u_id and m_id)
    '''

    def get_next_dataset(self, stage_id, types="only_new"):
        current_all_users, current_all_items, current_all_inter = [], set(), []
        try:
            # cold_users={}#leave-one-out:选择给定时间之后交互较少的用户作为cold-user，用于test-on-cold-users
            # current_stage_users=[]
            if types == "not_only_new":  # 将之前所有streaming的数据整合到一起作为历史数据
                score_data = pd.read_csv(self.path + '/streaming/' + str(stage_id) + '.dat',
                                         names=['user_id', 'item_id', 'rating', 'time'], engine='python')
                for i in range(0, stage_id):  # 将当前stage之前的数据
                    score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                    s_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating', 'time'],
                                         engine='python')
                    s_data = s_data.drop(0)
                    score_data = pd.concat([score_data, s_data])
                    score_data.reset_index(drop=True)
                warm_users, warm_users_data = self.split_warm_cold2(score_data)
                return warm_users, warm_users_data
            else:
                score_data_path = self.path + '/streaming/' + str(stage_id) + '.dat'
                score_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating', 'time'],
                                         engine='python')
                score_data = score_data.drop(0)
                warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(
                    score_data)  # cold and warm under current streaming

                # 选择在给定时间之后交互很少的用户作为cold-users（选择在当前streaming中交互很少的用户作为冷启动用户）
                return warm_users, cold_users, warm_users_data, cold_users_data
        except:
            print("read stage data wrong , may be there is no new data,finished")
            return None, None, None, None

    '''消融实验。获得full tune数据集，当下'''

    def get_current_full(self, stage_id):
        try:
            score_data = pd.read_csv(self.path + '/streaming/' + str(stage_id) + '.dat',
                                     names=['user_id', 'item_id', 'rating', 'time'],
                                     engine='python')
            users = score_data.user_id.unique()[1:]
            for i in range(0, stage_id):  # 将当前stage之前的数据
                score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                s_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating', 'time'],
                                     engine='python')
                score_data = pd.concat([score_data, s_data[s_data['user_id'].isin(users)]])
                score_data.reset_index(drop=True)
            warm_users, warm_users_data = self.split_warm_cold2(score_data)
            return warm_users, warm_users_data
        except:
            print("read stage data wrong , may be there is no new data,finished")
            return None, None, None, None

    def split_warm_cold2(self, rat_data, max_count=10):
        storing_path = './datasets'
        dataset = 'movielens'
        # 按照时间划分数据集
        sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        # print('start time %s,  end time %s' % (start_time, end_time))
        warm_users, user_counts = [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'time'])
        for u_id in user_ids:
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id] < 10 or u_count < 5: continue
            if u_id not in self.cold_users:
                warm_users.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
            user_counts.append(u_count)
        print('min count: %d, avg count: %d, max count: %d' % (
            min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))
        return warm_users, warm

    '''get all users and interaction and rating y based on interaction dataset score_data'''

    def get_inter_y(self, score_data):
        inters = []
        for idx, row in score_data.iterrows():
            if row['user_id'] == 'user_id':
                continue
            # if row['user_id'] not in users:
            #     users.append(row['user_id'])
            # current_all_items.add(row['movie_id'])
            inters.append([row['user_id'], row['item_id'], float(row['rating']), row['time']])

        users_movie_inter = collections.defaultdict(list)
        users_movie_inter_y = collections.defaultdict(list)
        for row in inters:  # 因为stage_data中的数据是按时间排序的
            # if row[-2]=='rating':continue#跳过表头
            users_movie_inter[row[0]].append(row[1])  # 直接取前80%就是时间上前80%的交互
            users_movie_inter_y[row[0]].append(row[-2])
        return users_movie_inter, users_movie_inter_y

    '''分场景，将每个streaming data中的users划分为support-set and query-set
    先对每个streaming data中的用户看过的电影，以及评分y

    随机选择前80%的交互作为supprot set，其余20%交互为query-set
    '''

    # not only new
    # test and val:choose the latest items as the test and validation instance following the leave-one-out evaluation
    # 根据时间预先划分streaming data
    def generate_current_suppAndQuery(self, stage_id, types="only_new"):
        # 基于get_next的数据集采样用户，生成这些用户的support-set and query-set
        # current_all_users, users_movie_inter, users_movie_inter_y = self.get_next_dataset(stage_id)
        warm_users, cold_users, warm_users_data, cold_users_data = self.get_next_dataset(stage_id, types="only_new")
        if not warm_users:
            print("***********read stage data wrong , may be there is no new data,finished***************")
            return None
        self.generate_warm_sq(stage_id, warm_users, warm_users_data, types="only_new")

        if cold_users:
            cold_inter, cold_inter_y = self.get_inter_y(cold_users_data)
            self.generate_cold_dataset(cold_users, cold_inter, cold_inter_y, stage_id)
        else:
            print("************this {} streaming has not cold users!***********".format(stage_id))
        # warm_users_full, warm_users_data_full = self.get_current_full(stage_id)
        # self.generate_warm_sq(stage_id, warm_users_full, warm_users_data_full, types="only_new_full")
        warm_users_f, warm_users_data_f = self.get_next_dataset(stage_id, types="not_only_new")
        self.generate_warm_sq(stage_id, warm_users_f, warm_users_data_f, types="not_only_new")
        print("**********generate dataset successfully******************")
        return True
        # 是写入文件还是直接返回suppor and query

    def generate_warm_sq(self, stage_id, warm_users, warm_users_data, types="only_new"):
        warm_inter, warm_inter_y = self.get_inter_y(warm_users_data)

        if types == "not_only_new":
            dir = "stage_dataset"
        elif types == "only_new_full":
            dir = "stage_full_tunning"
        else:
            dir = "stage_dataset_only_new"

        users = warm_users
        n = len(users)
        # generate the support set and query set
        idx = 0
        for u_id in users:  # 有时间顺序的
            support_x = []
            query_x = []
            seen_movie = len(warm_inter[u_id])
            indices = list(range(seen_movie))
            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set

            tmp_x = np.array(warm_inter[u_id])
            tmp_y = np.array(warm_inter_y[u_id])

            if seen_movie < 3: continue

            if seen_movie <= 5:
                s_user = pd.DataFrame([u_id] * floor(seen_movie * 0.8), columns=['user_id'])
                q_user = pd.DataFrame([u_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['item_id'])
                query_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['item_id'])
                support_x = s_user.join(support_x)
                query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])
            else:
                s_user = pd.DataFrame([u_id] * (seen_movie - 5), columns=['user_id'])
                q_user = pd.DataFrame([u_id] * 5, columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:-5]], columns=['item_id'])
                query_x = pd.DataFrame(tmp_x[indices[-5:]], columns=['item_id'])
                support_x = s_user.join(support_x)
                query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:-5]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[-5:]], columns=['rating'])

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            folder_path = "{}/{}/{}".format(self.path, dir, stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            support_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['user_id', 'item_id'], index=False)
            support_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['rating'], index=False)
            query_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['user_id', 'item_id'], index=False)
            query_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['rating'], index=False)
            idx += 1

    def get_cold_inter_y(self, cold_users_data):
        cu_movie_inter = collections.defaultdict(list)
        cu_movie_inter_y = collections.defaultdict(list)
        # each_stage_num=len(cold_user_idx)//stage_num
        for idx, row in cold_users_data.iterrows():  # 因为stage_data中的数据是按时间排序的
            cu_movie_inter[row['user_id']].append(row['item_id'])  # 直接取前80%就是时间上前80%的交互
            cu_movie_inter_y[row['user_id']].append(row['rating'])
        return cu_movie_inter, cu_movie_inter_y

    # 直接对每个cold
    def generate_cold_dataset(self, cold_user_idx, cu_movie_inter, cu_movie_inter_y, stage_id):
        idx = 0
        for cold_id in cold_user_idx:

            seen_movie = len(cu_movie_inter[cold_id])
            indices = list(range(seen_movie))

            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set
            tmp_x = np.array(cu_movie_inter[cold_id])
            tmp_y = np.array(cu_movie_inter_y[cold_id])
            train_user = pd.DataFrame([cold_id] * 5, columns=['user_id'])
            test_user = pd.DataFrame([cold_id] * 5, columns=['user_id'])

            train_x = pd.DataFrame(tmp_x[indices[:-5]], columns=['item_id'])
            train_y = pd.DataFrame(tmp_y[indices[:-5]], columns=['rating'])
            train_x = train_user.join(train_x)

            test_x = pd.DataFrame(tmp_x[indices[-5:]], columns=['item_id'])
            test_x = test_user.join(test_x)
            test_y = pd.DataFrame(tmp_y[indices[-5:]], columns=['rating'])

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            folder_path = "{}/{}/{}".format(self.path, 'cold_user', stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            train_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['user_id', 'item_id'], index=False)
            train_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['rating'], index=False)
            test_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['user_id', 'item_id'], index=False)
            test_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['rating'], index=False)
            idx += 1

    def item_converting(self, row, cate_list, title_list, price_list, brand_list):  # title and price 怎么编码
        # 'categories', 'title', 'price', 'brand'
        cate_idx = [0] * 1069
        for cate in row['categories'].split(", "):  # 有多个类目
            idx = cate_list.index(re.sub(r'\'', '',
                                         cate).strip())  # re.sub(r'\'', '', cate) str(re.sub(r'\'', '', cate).strip()) [re.sub(r'\'', '', cat).strip() for cat in set(c.split(','))]
            cate_idx[idx] = 1
        # cate_idx = cate_list.index(str(row['categories']))
        title_idx = title_list.index(str(row['title']))
        price_idx = price_list.index(str(row['price']))
        brand_idx = brand_list.index(str(row['brand']))
        out_list = list([title_idx, price_idx, brand_idx])
        out_list.extend(cate_idx)

        return out_list

    # def item_converting(self, row, rate_list, genre_list, director_list, year_list):
    #     # rate_dim: 6, year_dim: 1,  genre_dim:25, director_dim: 2186,
    #     rate_idx = rate_list.index(str(row['rate']))
    #     genre_idx = [0] * 25
    #     for genre in str(row['genre']).split(", "):
    #         idx = genre_list.index(genre)
    #         genre_idx[idx] = 1
    #     director_idx = [0] * 2186
    #     for director in str(row['director']).split(", "):
    #         idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
    #         director_idx[idx] = 1
    #     year_idx = year_list.index(row['year'])
    #     out_list = list([rate_idx, year_idx])
    #     out_list.extend(genre_idx)
    #     out_list.extend(director_idx)
    #     return out_list

    def load_list(self, fname):
        list_ = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f.readlines():
                list_.append(eval(line.strip()))
        return list_

    '''dataset = movielens_1m()
    item_data=dataset.item_data
    user_data=dataset.user_data
    '''

    def generate(self, master_path):  # 'categories', 'title', 'price', 'sales_type', 'sales_rank', 'brand'
        cate_list = self.load_list("{}/{}/categories.txt".format(self.path, '/row'))
        title_list = self.load_list("{}/{}/title.txt".format(self.path, '/row'))
        price_list = self.load_list("{}/{}/price.txt".format(self.path, '/row'))
        brand_list = self.load_list("{}/{}/brand.txt".format(self.path, '/row'))

        # hashmap for item information
        if not os.path.exists("{}/item_dict.pkl".format(master_path)):
            item_dict = {}
            for idx, row in self.item_data.iterrows():
                if row['item_id'] == 'item_id': continue  # 跳过表头
                m_info = self.item_converting(row, cate_list, title_list, price_list, brand_list)
                item_dict[row['item_id']] = m_info
            pickle.dump(item_dict, open("{}/item_dict.pkl".format(master_path), "wb"))
        else:
            item_dict = pickle.load(open("{}/item_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        return item_dict

    '''打开下一时刻的streaming data
    movielens/support/supp_x_stageid_{}.pkl    movielens/support/supp_y_stageid_{}.pkl
    movielens/query/query_x_stageid_{}.pkl     movielens/query/query_y_stageid_{}.pkl 
    x_stageid_{}.pkl  y_stageid_{}.pkl
    D_test=D_val
    一个stage_id对应一个streaming
    return: total_dataset
    根据dataset_path来生成warm and cold的数据集
    warm:datasets/movielens/stage_dataset        cold:datasets/movielens/cold_user
    '''

    # def next_dataset(self, stage_id, dataset_path, types="only_new",neg_num=0,sample_num=0):
    #     if types == "not_only_new":
    #         dir = "/stage_dataset/"
    #     elif types == "test_cold":
    #         dir = "/cold_user/"
    #     else:
    #         dir = "/stage_dataset_only_new/"
    #     dataset_path = dataset_path + dir + str(stage_id)
    #     try:  # get all filename
    #         data_file = os.listdir(dataset_path)
    #     except:
    #         print("filename error or filepath error!")
    #     all_tasks = [idx for idx in range(len(data_file) // 4)]
    #     if sample_num:
    #         # all_tasks = [idx for idx in range(len(data_file) // 4)]
    #         random.shuffle(all_tasks)
    #         all_tasks = all_tasks[:sample_num]
    #
    #     supp_x_s = []
    #     supp_y_s = []
    #     query_x_s = []
    #     query_y_s = []
    #     neg_x_s = []
    #     neg_y_s = []
    #     users_id = []
    #     for idx in all_tasks:
    #         s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
    #                           engine='python')
    #         s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
    #         q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
    #                           engine='python')
    #         q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
    #         support_x = [None, None]
    #         query_x = [None, None]
    #         support_y = None
    #         query_y = None
    #         u_id=None
    #         if len(s_x["user_id"]) < 1: continue
    #         inter_item = 0
    #         for idx, row in s_x.iterrows():
    #             if row['item_id'] == "item_id" or row['item_id'] not in self.item_dict.keys(): continue
    #             # support_x[0].append(self.u_dict[int(row['user_id'])])
    #             # support_x[1].append(self.m_dict[int(row['movie_id'])])
    #             # try:
    #             #     support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
    #             #                              0)  # 按维数0（行）拼接
    #             # except:
    #             #     support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
    #             # try:
    #             #     support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['item_id']]])),
    #             #                              0)  # 按维数0（行）拼接
    #             # except:
    #             #     support_x[1] = torch.tensor([self.m_dict[row['item_id']]])
    #             inter_item += 1
    #             u_id = row['user_id']
    #             s_x_converted = torch.tensor([self.item_dict[row['item_id']]])#按维数1（列）拼接
    #             try:
    #                 support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
    #             except:
    #                 support_x= s_x_converted
    #             s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
    #             try:
    #                 support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
    #             except:
    #                 support_y = s_y_converted
    #         if u_id:
    #             users_id.append(u_id)
    #         supp_x_s.append(support_x)
    #         supp_y_s.append(support_y)
    #
    #         for idx, row in q_x.iterrows():
    #             if row['item_id'] == "item_id" or row['item_id'] not in self.item_dict.keys(): continue
    #             # query_x[0].append(self.u_dict[int(row['user_id'])])
    #             # query_x[1].append(self.m_dict[int(row['movie_id'])])
    #             # try:
    #             #     query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
    #             #                            0)  # 按维数0（行）拼接
    #             # except:
    #             #     query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
    #             # try:
    #             #     query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['item_id']]])),
    #             #                            0)  # 按维数0（行）拼接
    #             # except:
    #             #     query_x[1] = torch.tensor([self.m_dict[row['item_id']]])
    #
    #             q_x_converted = torch.tensor([self.item_dict[row['item_id']]])#按维数1（列）拼接
    #             try:
    #                 query_x= torch.cat((query_x, q_x_converted), 0)#按维数0（行）拼接
    #             except:
    #                 query_x= q_x_converted
    #             q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
    #             try:
    #                 query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
    #             except:
    #                 query_y = q_y_converted
    #         query_x_s.append(query_x)
    #         query_y_s.append(query_y)
    #         if neg_num:
    #             neg_inter, neg_inter_y = self.gen_neg_train(s_x, q_x,
    #                                                         neg_number=inter_item)  # the number of neg_item is equal to pos_item number
    #             neg_x = [None]
    #             neg_y = [None]
    #             for idx, row in neg_inter.iterrows():
    #                 if row['movie_id'] == "movie_id": continue
    #                 try:
    #                     neg_x = torch.cat((neg_x, torch.tensor([self.item_dict[row['item_id']]])), 0)  # 按维数0（行）拼接
    #                 except:
    #                     neg_x = torch.tensor([self.item_dict[row['item_id']]])
    #                 s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
    #                 try:
    #                     neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
    #                 except:
    #                     neg_y = s_y_converted
    #             neg_x_s.append(neg_x)
    #             neg_y_s.append(neg_y)
    #
    #     if neg_num:
    #         d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
    #     else:
    #         d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
    #     return d_t, users_id
    def next_dataset(self, stage_id, dataset_path, types="only_new", neg_num=0):
        if types == "not_only_new":
            dir = "/stage_dataset/"
        elif types == "test_cold":
            dir = "/cold_user/"
        else:
            dir = "/stage_dataset_only_new/"
        dataset_path = dataset_path + dir + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        neg_x_s = []

        for idx in range((len(data_file) - 1) // 5):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['item_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['item_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = None
            query_x = None
            support_y = None
            query_y = None
            if len(s_x["item_id"]) < 1: continue
            inter_item = 0
            for idx, row in s_x.iterrows():
                if row['item_id'] == "item_id" or not row['item_id']: continue
                inter_item += 1

                s_x_converted = torch.tensor([self.item_dict[str(row['item_id'])]])  # 按维数1（列）拼接
                try:
                    support_x = torch.cat((support_x, s_x_converted), 0)  # 按维数0（行）拼接
                except:
                    support_x = s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)

            for idx, row in q_x.iterrows():
                if row['item_id'] == "item_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                # try:
                #     query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                #                            0)  # 按维数0（行）拼接
                # except:
                #     query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                # try:
                #     query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                #                            0)  # 按维数0（行）拼接
                # except:
                #     query_x[1] = torch.tensor([self.m_dict[row['item_id']]])

                q_x_converted = torch.tensor([self.item_dict[row['item_id']]])  # 按维数1（列）拼接
                try:
                    query_x = torch.cat((query_x, q_x_converted), 0)  # 按维数0（行）拼接
                except:
                    query_x = q_x_converted
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)
            if neg_num > 0:
                neg_inter, neg_inter_y = self.gen_neg_train(s_x, q_x,
                                                            neg_number=inter_item)  # the number of neg_item is equal to pos_item number
                neg_x = None
                neg_y = None
                for idx, row in neg_inter.iterrows():
                    if row['movie_id'] == "movie_id": continue
                    try:
                        neg_x = torch.cat((neg_x, torch.tensor([self.item_dict[row['item_id']]])), 0)  # 按维数0（行）拼接
                    except:
                        neg_x = torch.tensor([self.item_dict[row['item_id']]])
                    # s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
                    # try:
                    #     neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
                    # except:
                    #     neg_y = s_y_converted
                neg_x_s.append(neg_x)
                # neg_y_s.append(neg_y)
            # support_x = pickle.load(open('{}/supp_x_{}.pkl'.format(dataset_path, idx), 'rb'))
            # support_y = pickle.load(open('{}/supp_y_{}.pkl'.format(dataset_path, idx), 'rb'))
            # query_x = pickle.load(open('{}/query_x_{}.pkl'.format(dataset_path, idx), 'rb'))
            # query_y = pickle.load(open('{}/query_y_{}.pkl'.format(dataset_path, idx), 'rb'))
            #
            # if neg_num:
            #     neg_x = pickle.load(open('{}/neg_x_{}.pkl'.format(dataset_path, idx), 'rb'))
            #     neg_x_s.append(neg_x)
            # supp_x_s.append(support_x)
            # supp_y_s.append(support_y)
            # query_x_s.append(query_x)
            # query_y_s.append(query_y)

        users_id = eval(open('{}/users_id.txt'.format(dataset_path)).read())
        if neg_num > 0:
            d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
        else:
            d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t, users_id

        # 选择负样本，直接拼接到训练集末尾

    def gen_neg_train(self, supx_set, qurx_set, neg_number=None):
        if neg_number is None:
            print("dont't select neg sample numbers!!")
            return None
        item_ids = self.item_data.movie_id.unique()
        random.shuffle(item_ids)
        all_data = pd.concat([supx_set, qurx_set], axis=0)
        inter_items = all_data.movie_id.unique()

        neg_inter = pd.DataFrame(columns=['user_id', "movie_id"])
        neg_inter_y = pd.DataFrame(columns=['rating'])
        for item in item_ids:
            if item not in inter_items:
                neg_inter = neg_inter.append(
                    pd.DataFrame([[supx_set.iloc[-1].user_id, item]], columns=['user_id', 'movie_id']))
                # neg_inter = neg_inter.append(neg_inter)
                neg_inter_y = neg_inter_y.append(pd.DataFrame([0], columns=['rating']))
                neg_number -= 1
                if neg_number <= 0: break
        return neg_inter.reset_index(drop=True), neg_inter_y.reset_index(drop=True)

    # add new users to history dataset after each roung of training
    def add_new_user_to_history(self, stage_id, types="only_new"):
        if types == "not_only_new":
            history_path = self.path + "/stage_dataset/{}".format(stage_id)
        else:
            history_path = self.path + "/stage_dataset_only_new/{}".format(stage_id)
        cold_user_path = self.path + "/cold_user/{}".format(stage_id)
        max_idx = len(os.listdir(history_path)) // 4

        for idx in range(len(os.listdir(cold_user_path)) // 4):
            train_x = pd.read_csv(cold_user_path + '/supp_x_{}.dat'.format(idx), names=['user_id', 'item_id'],
                                  engine='python')
            train_y = pd.read_csv(cold_user_path + '/supp_y_{}.dat'.format(idx), names=['rating'], engine='python')
            test_x = pd.read_csv(cold_user_path + '/query_x_{}.dat'.format(idx), names=['user_id', 'item_id'],
                                 engine='python')
            test_y = pd.read_csv(cold_user_path + '/query_y_{}.dat'.format(idx), names=['rating'], engine='python')

            # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            train_x.to_csv("{}/supp_x_{}.dat".format(history_path, idx + max_idx), header=None,
                           columns=['user_id', 'item_id'], index=False)
            train_y.to_csv("{}/supp_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'],
                           index=False)
            test_x.to_csv("{}/query_x_{}.dat".format(history_path, idx + max_idx), header=None,
                          columns=['user_id', 'item_id'], index=False)
            test_y.to_csv("{}/query_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'],
                          index=False)
        # shutil.rmtree(cold_user_path)

    def next_full_dataset(self, stage_id, dataset_path):
        path = "/stage_full_tunning/"
        dataset_path = dataset_path + path + str(stage_id)
        try:  # get all filename
            data_file = os.listdir(dataset_path)
        except:
            print("filename error or filepath error!")
            return
        supp_x_s = []
        supp_y_s = []
        query_x_s = []
        query_y_s = []
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],
                              engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['item_id'] == "item_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    support_x[1] = torch.cat((support_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[1] = torch.tensor([self.m_dict[row['item_id']]])
                # s_x_converted = torch.cat((self.m_dict[int(row['item_id'])], self.u_dict[int(row['user_id'])]), 1)#按维数1（列）拼接
                # try:
                #     support_x= torch.cat((support_x, s_x_converted), 0)#按维数0（行）拼接
                # except:
                #     support_x= s_x_converted
                s_y_converted = torch.tensor([float(s_y.iloc[idx]['rating'])])
                try:
                    support_y = torch.cat((support_y, s_y_converted), 0)  # 按维数0（行）拼接
                except:
                    support_y = s_y_converted

            supp_x_s.append(support_x)
            supp_y_s.append(support_y)
            for idx, row in q_x.iterrows():
                if row['item_id'] == "item_id": continue
                # query_x[0].append(self.u_dict[int(row['user_id'])])
                # query_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
                try:
                    query_x[1] = torch.cat((query_x[1], torch.tensor([self.m_dict[row['item_id']]])),
                                           0)  # 按维数0（行）拼接
                except:
                    query_x[1] = torch.tensor([self.m_dict[row['item_id']]])
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

        d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t


if __name__ == "__main__":
    # data_path = '../dataset/'  # /home/ly525999/MEPM
    # data_name = 'news_3week_all'
    # file_list = ["2017010" + str(i) if i < 10 else "201701" + str(i) for i in range(1, 22)]
    # select_neg_forinteraction(path=data_path, datasetname=data_name, file_path_list=file_list)
    stage_num = 50  # 大约每五天更新一次
    states = ['warm', 'cold_user']
    ml_dataset = online_movielens_1m()  # states,stage_num)
