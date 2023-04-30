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

'''
dataset=online_movielens_1m(stage_num)
to generate next streaming support set and query set 
self.get_streaming_interAndY(stage_id=,dataset.u_dict,dtaset.m_dict)
to get stageid support and query set with open file
movielens/support/supp_x_stageid_第几个用户.pkl
'''


class online_movielens_1m(object):
    def __init__(self, stage_num=45):
        self.path = '../datasets/movielens'
        self.user_data, self.item_data, self.score_data = self.load()
        self.user_num, self.item_num = len(self.user_data) - 1, len(self.item_data)

        self.u_dict, self.m_dict = self.generate(self.path, self.item_data, self.user_data)
        # warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num,warm_users_data)
        #self.generate_streaming(stage_num, self.score_data)
        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()
        # # # #
        # # # # #generate each time(stage) support and query set
        for i in tqdm(range(stage_num)):
            self.generate_current_suppAndQuery(i, types="only_new")


    # load all users and items and rating in row data
    def load(self):

        path = "../datasets/movielens/ml-1m"
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
        storing_path = '../datasets'
        dataset = 'movielens'
        # 按照时间划分数据集
        sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time,  end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        print('start time %s,  end time %s' % (start_time, end_time))
        user_warm_list, user_cold_list, user_counts = [], [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm, cold = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'time']), pd.DataFrame(
            columns=['user_id', 'movie_id', 'rating', 'time'])
        for u_id in user_ids:
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id]<2 or u_count<2:continue
            if 5<self.user_current_interNum[u_id] <= 30:
                user_cold_list.append(u_id)
                self.cold_users.append(u_id)
                self.cold_users_inter = pd.concat([self.cold_users_inter,u_info]).reset_index(drop=True)

            else:
                user_warm_list.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
                if u_id in self.cold_users:
                    self.cold_users.remove(u_id)
                    warm = pd.concat([warm, self.cold_users_inter[self.cold_users_inter['user_id'] == u_id]]).reset_index(drop=True)
                    self.cold_users_inter.drop(self.cold_users_inter[self.cold_users_inter['user_id'] == u_id].index)
            user_counts.append(u_count)
        #cold = pd.concat([cold, u_info]).reset_index(drop=True)  # 取出目前为止该用户的所有交互

        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('min count: %d, avg count: %d, max count: %d' % (min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
                          'user_cold_ids': user_cold_list}
        #pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
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
                                         names=['user_id', 'movie_id', 'rating', 'time'], engine='python')
                for i in range(0, stage_id):  # 将当前stage之前的数据
                    score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                    s_data = pd.read_csv(score_data_path, names=['user_id', 'movie_id', 'rating', 'time'],
                                         engine='python')
                    score_data = pd.concat([score_data, s_data])
                    score_data.reset_index(drop=True)
                warm_users,  warm_users_data= self.split_warm_cold2(score_data)
                return warm_users, warm_users_data
            else:
                score_data_path = self.path + '/streaming/' + str(stage_id) + '.dat'
                score_data = pd.read_csv(score_data_path, names=['user_id', 'movie_id', 'rating', 'time'],
                                         engine='python')
                warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(score_data)  # cold and warm under current streaming

                # 选择在给定时间之后交互很少的用户作为cold-users（选择在当前streaming中交互很少的用户作为冷启动用户）
                return warm_users, cold_users, warm_users_data, cold_users_data
        except:
            print("read stage data wrong , may be there is no new data,finished")
            return None, None, None, None

    '''消融实验。获得full tune数据集，当下'''
    def get_current_full(self, stage_id):
        try:
            score_data = pd.read_csv(self.path + '/streaming/' + str(stage_id) + '.dat', names=['user_id', 'movie_id', 'rating', 'time'],
                                     engine='python')
            users = score_data.user_id.unique()[1:]
            for i in range(0, stage_id):  # 将当前stage之前的数据
                score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                s_data = pd.read_csv(score_data_path, names=['user_id', 'movie_id', 'rating', 'time'],
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
        start_time,  end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        #print('start time %s,  end time %s' % (start_time, end_time))
        warm_users, user_counts = [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'time'])
        for u_id in user_ids:
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if u_id not in self.cold_users:
                warm_users.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
            user_counts.append(u_count)
        print('min count: %d, avg count: %d, max count: %d' % (min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))
        return warm_users,  warm

    '''get all users and interaction and rating y based on interaction dataset score_data'''
    def get_inter_y(self, score_data):
        inters = []
        for idx, row in score_data.iterrows():
            if row['user_id'] == 'user_id':
                continue
            # if row['user_id'] not in users:
            #     users.append(row['user_id'])
            #current_all_items.add(row['movie_id'])
            inters.append([row['user_id'], row['movie_id'], float(row['rating']), row['time']])

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
        #current_all_users, users_movie_inter, users_movie_inter_y = self.get_next_dataset(stage_id)
        warm_users, cold_users, warm_users_data, cold_users_data = self.get_next_dataset(stage_id, types="only_new")
        if not warm_users:
            print("***********read stage data wrong , may be there is no new data,finished***************")
            return None
        self.generate_warm_sq(stage_id,warm_users, warm_users_data, types="only_new")

        if cold_users:
            cold_inter, cold_inter_y = self.get_inter_y(cold_users_data)
            self.generate_cold_dataset(cold_users, cold_inter, cold_inter_y, stage_id)
        else:
            print("************this {} streaming has not cold users!***********".format(stage_id))
        warm_users_full, warm_users_data_full = self.get_current_full(stage_id)
        self.generate_warm_sq(stage_id, warm_users_full, warm_users_data_full, types="only_new_full")
        warm_users_f, warm_users_data_f = self.get_next_dataset(stage_id, types="not_only_new")
        self.generate_warm_sq(stage_id, warm_users_f, warm_users_data_f, types="not_only_new")
        print("**********generate dataset successfully******************")
        return True
        # 是写入文件还是直接返回suppor and query
    def generate_warm_sq(self, stage_id, warm_users, warm_users_data, types = "only_new"):
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

            if seen_movie<=5:
                s_user = pd.DataFrame([u_id] * floor(seen_movie * 0.8), columns=['user_id'])
                q_user = pd.DataFrame([u_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['movie_id'])
                query_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['movie_id'])
                support_x = s_user.join(support_x)
                query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])
            else:
                s_user = pd.DataFrame([u_id] * (seen_movie - 5), columns=['user_id'])
                q_user = pd.DataFrame([u_id] * 5, columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:-5]], columns=['movie_id'])
                query_x = pd.DataFrame(tmp_x[indices[-5:]], columns=['movie_id'])
                support_x = s_user.join(support_x)
                query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:-5]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[-5:]], columns=['rating'])

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
            movie_dict = {}
            for idx, row in item_data.iterrows():
                if row['movie_id'] == 'movie_id': continue  # 跳过表头
                m_info = self.item_converting(row, rate_list, genre_list, director_list, year_list)
                movie_dict[row['movie_id']] = m_info
            pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            user_dict = {}
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
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'], engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'], engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
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

        d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t

    # add new users to history dataset after each roung of training
    def add_new_user_to_history(self, stage_id, types="only_new"):
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
            train_y.to_csv("{}/supp_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'], index=False)
            test_x.to_csv("{}/query_x_{}.dat".format(history_path, idx + max_idx), header=None,
                          columns=['user_id', 'movie_id'], index=False)
            test_y.to_csv("{}/query_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'], index=False)
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
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
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
                q_y_converted = torch.tensor([float(q_y.iloc[idx]['rating'])])
                try:
                    query_y = torch.cat((query_y, q_y_converted), 0)  # 按维数0（行）拼接
                except:
                    query_y = q_y_converted
            query_x_s.append(query_x)
            query_y_s.append(query_y)

        d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t


'''
online Amazon dataset
split interactions into 50 periods.
'''
class online_amazon(object):
    def __init__(self, stage_num=30):
        self.path = '../datasets/amazon'
        # self.item_data, self.score_data = self.load()
        # user_ids = list(set(self.score_data.user_id))
        # self.user_num, self.item_num = self.score_data.user_id.nunique(), self.item_data.item_id.nunique()
        # self.gen_attribution(self.item_data)

        # self.split_warm_cold(self.score_data)

        self.item_dict = self.generate(self.path)
        # warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num,warm_users_data)

        # self.generate_streaming(stage_num, self.score_data)
        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()
        self.few_inter = pd.DataFrame()
        # # # # #generate each time(stage) support and query set
        for i in tqdm(range(stage_num)):
          self.generate_current_suppAndQuery(i, types="only_new")

    # load all users and items and rating in row data
    def load(self):
        path = "../datasets/amazon/row"
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

        # item_data = item_data.drop(columns=['sales_rank', 'sales_type'])

        # item_data=item_data[item_data['price'].notnull()]
        # item_data = item_data[item_data['title'].notnull()]
        # item_data = item_data[item_data['sales_rank'].notnull()]
        # item_data = item_data[item_data['sales_type'].notnull()]

        item_data['title'].fillna(method='bfill', inplace=True)
        price = item_data[item_data['price'].notnull()].price.tolist()
        avg_p = round(sum([float(i) for i in price]) / len(price), 2)
        item_data['price'].fillna(str(avg_p), inplace=True)
        item_data['brand'].fillna(method='bfill', inplace=True)  # ffill:以前一个非NAN值填充

        item_data['sales_type'].fillna('Electronics', inplace=True)

        # user_id:token	item_id:token	rating:float	timestamp:float
        score_data = pd.read_csv(
            inter_data_path, names=['user_id', 'item_id', 'rating', 'timestamp'],
            sep="\t", engine='python'
        )
        score_data = score_data.drop(0)

        # score_data=self.filter_data(score_data)

        # for idx, row in score_data.iterrows():
        #     if row['item_id'] not in item_data.item_id.unique():
        #         score_data.drop(idx,inplace=True)
        # score_data = score_data.drop(score_data.index(score_data['item_id'] not in item_data.item_id.unique()))
        score_data = score_data.sort_values(by=['timestamp'])  # 按时间戳升序

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(int(x)))
        score_data = score_data.drop(["timestamp"], axis=1)
        return item_data, score_data

    def filter_data(self,score):
        all_user = score.user_id.unique()
        for user in all_user:
            if len(score[score['user_id']==user])<10:
                score.drop(score.index[(score['user_id']==user)],inplace=True)
        return score

    def gen_attribution(self, item_data):#'categories', 'title', 'price', 'sales_type', 'sales_rank', 'brand'
        file_path = self.path + "/row/"

        categories_list = item_data.categories.unique()
        title_list = item_data.title.unique()
        price_list = item_data.price.unique()
        brand_list = item_data.brand.unique()
        sales_type = item_data.sales_type.unique()
        sales_rank = item_data.sales_rank.unique()
        self.write_attr(file_path + 'categories.txt', categories_list)
        self.write_attr(file_path + 'title.txt', title_list)
        self.write_attr(file_path + 'price.txt', price_list)
        self.write_attr(file_path + 'brand.txt', brand_list)
        self.write_attr(file_path + 'type.txt', sales_type)
        self.write_attr(file_path + 'rank.txt', sales_rank)

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
        storing_path = '../datasets'
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
            if self.user_current_interNum[u_id]<7 or u_count<6:continue

            if self.user_current_interNum[u_id] <= 20 and len(self.cold_users)<=200:
                # user_cold_list.append(u_id)
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

        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(self.cold_users)))
        print('min count: %d, max count: %d, sum count: %d, average count: %d' % (min(user_counts), max(user_counts),sum(user_counts),sum(user_counts)//len(user_counts)))

        new_all_ids = new_df.user_id.unique()

        # user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
        #                   'user_cold_ids': user_cold_list}
        # pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
        return user_warm_list, self.cold_users, warm, self.cold_users_inter  # pd.DataFrame(warm,columns=['user_id', 'movie_id', 'rating', 'time']),pd.DataFrame(cold,columns=['user_id', 'movie_id', 'rating', 'time'],)


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
                score_data=score_data.drop(0)
                warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(score_data)  # cold and warm under current streaming

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
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id] < 10 or self.user_current_interNum[u_id]>500: continue
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


        # warm_users_f, warm_users_data_f = self.get_next_dataset(stage_id, types="not_only_new")
        # self.generate_warm_sq(stage_id, warm_users_f, warm_users_data_f, types="not_only_new")
        print("**********generate dataset successfully******************")
        return True
        # 是写入文件还是直接返回suppor and query

    def generate_warm_sq(self, stage_id, warm_users, warm_users_data, types="only_new",neg_num=10):
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

            if seen_movie<5:continue

            if seen_movie <= 10:
                # s_user = pd.DataFrame([u_id] * floor(seen_movie * 0.8), columns=['user_id'])
                # q_user = pd.DataFrame([u_id] * (seen_movie - floor(seen_movie * 0.8)), columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:floor(seen_movie * 0.8)]], columns=['item_id'])
                query_x = pd.DataFrame(tmp_x[indices[floor(seen_movie * 0.8):]], columns=['item_id'])
                # support_x = s_user.join(support_x)
                # query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:floor(seen_movie * 0.8)]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[floor(seen_movie * 0.8):]], columns=['rating'])
            else:
                # s_user = pd.DataFrame([u_id] * (seen_movie - 5), columns=['user_id'])
                # q_user = pd.DataFrame([u_id] * 5, columns=['user_id'])

                support_x = pd.DataFrame(tmp_x[indices[:-5]], columns=['item_id'])
                query_x = pd.DataFrame(tmp_x[indices[-5:]], columns=['item_id'])
                # support_x = s_user.join(support_x)
                # query_x = q_user.join(query_x)
                support_y = pd.DataFrame(tmp_y[indices[:-5]], columns=['rating'])
                query_y = pd.DataFrame(tmp_y[indices[-5:]], columns=['rating'])
            folder_path = "{}/{}/{}".format(self.path, dir, stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            support_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['item_id'], index=False)
            support_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                             columns=['rating'], index=False)
            query_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['item_id'], index=False)
            query_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, dir, stage_id, idx), header=None,
                           columns=['rating'], index=False)
            idx += 1

        f = open("{}/{}/{}/users_id.txt".format(self.path, dir, stage_id), "w")
        f.write(str(users))
        f.close()

        #     s_x, q_x, s_y, q_y = None, None, None, None
        #
        #     if len(support_x["item_id"]) < 1: continue
        #     inter_item = 0
        #     for idx, row in support_x.iterrows():
        #         if row['item_id'] == "item_id" or not row['item_id']: continue
        #         inter_item += 1
        #         s_x_converted = torch.tensor([self.item_dict[row['item_id']]])  # 按维数1（列）拼接
        #         try:
        #             s_x = torch.cat((s_x, s_x_converted), 0)  # 按维数0（行）拼接
        #         except:
        #             s_x = s_x_converted
        #         s_y_converted = torch.tensor([support_y['rating']])
        #         try:
        #             s_y = torch.cat((s_y, s_y_converted), 0)  # 按维数0（行）拼接
        #         except:
        #             s_y = s_y_converted
        #
        #     for idx, row in query_x.iterrows():
        #         if row['item_id'] == "item_id" or not row['item_id']: continue
        #
        #         q_x_converted = torch.tensor([self.item_dict[row['item_id']]])  # 按维数1（列）拼接
        #         try:
        #             q_x = torch.cat((q_x, q_x_converted), 0)  # 按维数0（行）拼接
        #         except:
        #             q_x = q_x_converted
        #         q_y_converted = torch.tensor([query_y['rating']])
        #         try:
        #             q_y = torch.cat((q_y, q_y_converted), 0)  # 按维数0（行）拼接
        #         except:
        #             q_y = q_y_converted
        #
        #     folder_path = "{}/{}/{}".format(self.path, dir, stage_id)
        #     if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        #         os.makedirs(folder_path)
        #     if neg_num:
        #         neg_inter= self.gen_neg_train(support_x, query_x,neg_number=inter_item)  # the number of neg_item is equal to pos_item number
        #         neg_x = [None]
        #         # neg_y = [None]
        #         for idx, row in neg_inter.iterrows():
        #             if row['item_id'] == "item_id": continue
        #             try:
        #                 neg_x = torch.cat((neg_x, torch.tensor([self.item_dict[row['item_id']]])), 0)  # 按维数0（行）拼接
        #             except:
        #                 neg_x = torch.tensor([self.item_dict[row['item_id']]])
        #             # s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
        #             # try:
        #             #     neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
        #             # except:
        #             #     neg_y = s_y_converted
        #         pickle.dump(neg_x,
        #                     open("{}/neg_x_{}.pkl".format(folder_path, idx),
        #                          'wb'))
        #         # pickle.dump(neg_y,
        #         #             open("{}/neg_y_{}.pkl".format(folder_path, idx),
        #         #                  'wb'))
        #
        #
        #     # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
        #
        #     pickle.dump(s_x, open("{}/supp_x_{}.pkl".format(folder_path, idx), 'wb'))
        #     pickle.dump(s_y, open("{}/supp_y_{}.pkl".format(folder_path, idx), 'wb'))
        #     pickle.dump(q_x, open("{}/query_x_{}.pkl".format(folder_path, idx), 'wb'))
        #     pickle.dump(q_y, open("{}/query_y_{}.pkl".format(folder_path, idx), 'wb'))
        #     idx += 1


    def get_cold_inter_y(self, cold_users_data):
        cu_movie_inter = collections.defaultdict(list)
        cu_movie_inter_y = collections.defaultdict(list)
        # each_stage_num=len(cold_user_idx)//stage_num
        for idx, row in cold_users_data.iterrows():  # 因为stage_data中的数据是按时间排序的
            cu_movie_inter[row['user_id']].append(row['item_id'])  # 直接取前80%就是时间上前80%的交互
            cu_movie_inter_y[row['user_id']].append(row['rating'])
        return cu_movie_inter, cu_movie_inter_y

    # 直接对每个cold
    def generate_cold_dataset(self, cold_user_idx, cu_movie_inter, cu_movie_inter_y, stage_id, neg_num=1):
        idx = 0
        for cold_id in cold_user_idx:

            seen_movie = len(cu_movie_inter[cold_id])
            indices = list(range(seen_movie))

            random.shuffle(indices)  # 打乱interactions的顺序，以保证随机采样生成support set
            tmp_x = np.array(cu_movie_inter[cold_id])
            tmp_y = np.array(cu_movie_inter_y[cold_id])
            # train_user = pd.DataFrame([cold_id] * 5, columns=['user_id'])
            # test_user = pd.DataFrame([cold_id] * 5, columns=['user_id'])

            train_x = pd.DataFrame(tmp_x[indices[:-5]], columns=['item_id'])
            train_y = pd.DataFrame(tmp_y[indices[:-5]], columns=['rating'])
            # train_x = train_user.join(train_x)

            test_x = pd.DataFrame(tmp_x[indices[-5:]], columns=['item_id'])
            # test_x = test_user.join(test_x)
            test_y = pd.DataFrame(tmp_y[indices[-5:]], columns=['rating'])
            folder_path = "{}/{}/{}".format(self.path, 'cold_user', stage_id)
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)
            train_x.to_csv("{}/{}/{}/supp_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['item_id'], index=False)
            train_y.to_csv("{}/{}/{}/supp_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                           columns=['rating'], index=False)
            test_x.to_csv("{}/{}/{}/query_x_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['item_id'], index=False)
            test_y.to_csv("{}/{}/{}/query_y_{}.dat".format(self.path, 'cold_user', stage_id, idx), header=None,
                          columns=['rating'], index=False)
            idx += 1

            # s_x, q_x, s_y, q_y = None, None, None, None
            #
            # if len(train_x["item_id"]) < 1: continue
            # inter_item = 0
            # for idx, row in train_x.iterrows():
            #     if row['item_id'] == "item_id" or not row['item_id']: continue
            #     inter_item += 1
            #     s_x_converted = torch.tensor([self.item_dict[row['item_id']]])  # 按维数1（列）拼接
            #     try:
            #         s_x = torch.cat((s_x, s_x_converted), 0)  # 按维数0（行）拼接
            #     except:
            #         s_x = s_x_converted
            #     s_y_converted = torch.tensor(list(train_y['rating']))
            #     try:
            #         s_y = torch.cat((s_y, s_y_converted), 0)  # 按维数0（行）拼接
            #     except:
            #         s_y = s_y_converted
            #
            # for idx, row in test_x.iterrows():
            #     if row['item_id'] == "item_id" or not row['item_id']: continue
            #
            #     q_x_converted = torch.tensor([self.item_dict[row['item_id']]])  # 按维数1（列）拼接
            #     try:
            #         q_x = torch.cat((q_x, q_x_converted), 0)  # 按维数0（行）拼接
            #     except:
            #         q_x = q_x_converted
            #     q_y_converted = torch.tensor(list(test_y['rating']))
            #     try:
            #         q_y = torch.cat((q_y, q_y_converted), 0)  # 按维数0（行）拼接
            #     except:
            #         q_y = q_y_converted
            # folder_path = "{}/{}/{}".format(self.path, 'cold_user', stage_id)
            # if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder_path)
            # if neg_num:
            #     neg_inter = self.gen_neg_train(train_x, test_x,neg_number=inter_item)  # the number of neg_item is equal to pos_item number
            #     neg_x = [None]
            #
            #     for idx, row in neg_inter.iterrows():
            #         if row['item_id'] == "item_id": continue
            #         try:
            #             neg_x = torch.cat((neg_x, torch.tensor([self.item_dict[row['item_id']]])), 0)  # 按维数0（行）拼接
            #         except:
            #             neg_x = torch.tensor([self.item_dict[row['item_id']]])
            #         # s_y_converted = torch.tensor([float(neg_inter_y.iloc[idx]['rating'])])
            #         # try:
            #         #     neg_y = torch.cat((neg_y, s_y_converted), 0)  # 按维数0（行）拼接
            #         # except:
            #         #     neg_y = s_y_converted
            #     pickle.dump(neg_x, open("{}/neg_x_{}.pkl".format(folder_path,  idx), 'wb'))
            #     # pickle.dump(neg_y, open("{}/neg_y_{}.pkl".format(folder_path, idx), 'wb'))
            #
            # # data_path:movielens/support/supp_x_stageid_第几个用户.pkl
            #
            # pickle.dump(s_x, open("{}/supp_x_{}.pkl".format(folder_path, idx), 'wb'))
            # pickle.dump(s_y, open("{}/supp_y_{}.pkl".format(folder_path, idx), 'wb'))
            # pickle.dump(q_x, open("{}/query_x_{}.pkl".format(folder_path, idx), 'wb'))
            # pickle.dump(q_y, open("{}/query_y_{}.pkl".format(folder_path, idx), 'wb'))
            #
            # idx += 1
        f = open("{}/{}/{}/users_id.txt".format(self.path, "cold_user", stage_id), "w")
        f.write(str(cold_user_idx))
        f.close()

    def gen_neg_train(self, supx_set, qurx_set, neg_number=None):
        if neg_number is None:
            print("dont't select neg sample numbers!!")
            return None
        item_ids = self.item_data.item_id.unique()
        random.shuffle(item_ids)
        all_data = pd.concat([supx_set, qurx_set], axis=0)
        inter_items = all_data.item_id.unique()

        neg_inter = pd.DataFrame(columns=["item_id"])
        neg_inter_y = pd.DataFrame(columns=['rating'])
        for item in item_ids:
            if item not in inter_items:
                neg_inter = neg_inter.append(
                    pd.DataFrame([item], columns=['item_id']))
                # neg_inter = neg_inter.append(neg_inter)
                neg_inter_y = neg_inter_y.append(pd.DataFrame([0], columns=['rating']))
                neg_number -= 1
                if neg_number <= 0: break
        return neg_inter.reset_index(drop=True)

    def item_converting(self, row, cate_list, title_list, price_list, brand_list):  # title and price 怎么编码
        # 'categories', 'title', 'price', 'brand'
        cate_idx = [0] * 1318
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
                list_.append(line.strip())
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
                # if row['item_id']=='B00007IT9B':print("************************")
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

    def next_dataset(self, stage_id, dataset_path, types="only_new",neg_num=0):
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

        for idx in range((len(data_file)-1) // 5):
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

                s_x_converted = torch.tensor([self.item_dict[row['item_id']]])  # 按维数1（列）拼接
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
            if neg_num:
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
        if neg_num:
            d_t = list(zip(supp_x_s, supp_y_s, neg_x_s, query_x_s, query_y_s))
        else:
            d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t, users_id

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



class online_yelp(object):
    def __init__(self, stage_num=40):
        self.path = '../datasets/yelp'
        self.user_data, self.item_data, self.score_data = self.load()
        #user_ids = list(set(self.score_data.user_id))
        self.user_num, self.item_num = self.userdata.user_id.nunique(), self.item_data.item_id.nunique()

        # self.split_warm_cold(self.score_data)

        # self.m_dict = self.generate(self.path, self.item_data)
        # warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num,warm_users_data)

        self.generate_streaming(stage_num, self.score_data)
        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()
        # # # # #generate each time(stage) support and query set
        # for i in tqdm(range(stage_num)):
        #     self.generate_current_suppAndQuery(i, types="only_new")

    # load all users and items and rating in row data
    def load(self):
        path = "../datasets/yelp/row"
        user_data_path = "{}/yelp.user".format(path)
        inter_data_path = "{}/yelp.inter".format(path)
        item_data_path = "{}/yelp.item".format(path)

        user_data = pd.read_csv(user_data_path, names=['user_id', 'user_name', 'user_review_count', 'yelping_since', 'user_useful', 'user_funny', 'user_cool','elite','fans','average_stars', 'compliment_hot','compliment_more','compliment_profile','compliment_cute','compliment_list','compliment_note','compliment_plain','compliment_cool','compliment_funny','compliment_writer','compliment_photos'],
                                   sep="\t", engine='python')
        user_data = user_data.drop(0)

        #删除一些无关列
        # 可以根据user_review_count筛选用户
        user_data=user_data.drop(columns=['user_name'])

        item_data = pd.read_csv(
            item_data_path,
            names=['item_id', 'item_name', 'address', 'city', 'state', 'postal_code', 'latitude','longitude','item_stars','item_review_count','is_open','categories'],
            sep="\t", engine='python')
        item_data = item_data.drop(0)
        # item_data = item_data.dropna()
        item_data = item_data.drop(columns=['item_name'])

        #review_id:token	user_id:token	business_id:token	stars:float	useful:float	funny:float	cool:float	date:float
        score_data = pd.read_csv(
            inter_data_path, names=['review_id', 'user_id', 'item_id', 'stars','useful','funny','cool','timestamp'],
            sep="\t", engine='python')
        score_data = score_data.drop(0)
        score_data = score_data.drop(columns=['review_id'])
        score_data = score_data.sort_values(by=['timestamp'])  # 按时间戳升序
        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(int(x)))
        score_data = score_data.drop(["timestamp"], axis=1)
        score_data = self.filter_data(score_data)
        return user_data,item_data, score_data

    def filter_data(self,score):
        all_user = score.user_id.unique()
        for user in all_user:
            if len(score[score['user_id']==user])<20:
                score.drop(score.index[(score['user_id']==user)],inplace=True)
        return score


    def gen_attribution(self, user_data, item_data):
        file_path = self.path + "/row/"
        age_list = user_data.age.unique()
        location_list = user_data.location.unique()
        self.write_attr(file_path + 'b_age.txt', age_list)
        self.write_attr(file_path + 'b_location.txt', location_list)

        author_list = item_data.author.unique()
        publisher_list = item_data.publisher.unique()
        self.write_attr(file_path + 'b_author.txt', author_list)
        self.write_attr(file_path + 'b_publisher.txt', publisher_list)

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
        storing_path = '../datasets'
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
            if self.user_current_interNum[u_id] < 2 or u_count < 2: continue
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
                score_data=score_data.drop(0)
                warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(score_data)  # cold and warm under current streaming

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
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
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
            movie_dict = {}
            for idx, row in item_data.iterrows():
                if row['item_id'] == 'item_id': continue  # 跳过表头
                m_info = self.item_converting(row, rate_list, genre_list, director_list, year_list)
                movie_dict[row['item_id']] = m_info
            pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            user_dict = {}
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
        return d_t

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



# class online_yelp:
#
#     def __init__(self, path="dataset/", datasetname="yelp", file_path_list=None, test_list=None, validation_list=None,
#                  partion=7 / 21, test_partion=16 / 21, need_pbar=False):
#         self.path = path
#         self.dataname = datasetname
#         self.file_list = file_path_list
#         self.test_list = test_list
#         self.val_list = validation_list
#         self.len = len(file_path_list)
#         self.start_time = round(self.len * partion)  # start not use as train
#         self.start_test_time = round(self.len * test_partion)  # start as test
#         self.test_count = 0
#         self.train_count = 0
#         information = np.load(self.path + self.dataname + "/" + "information.npy")
#         self.user_number = information[1]
#         self.inter_all = information[0]
#         self.item_number = information[2]
#         print(information)
#         a = [np.load(self.path + self.dataname + "/train/" + file + ".npy") for file in self.file_list]
#         a = np.concatenate(a, axis=0)
#         print(a.shape[0], np.unique(a[:, 0]).shape[0], np.unique(a[:, 1]).shape[0])
#
#     def get_offline(self, do_test=True, do_val=True):
#         '''
#         :return: offline train set and list of test set
#         '''
#
#         test_set = None
#         val_set = None
#         if self.test_list is not None and do_test:
#             test_set = [np.load(self.path + self.dataname + "/test/" + test_file + ".npy") for test_file in
#                         self.test_list]
#             print("the number of test file is:", len(test_set))
#         if self.val_list is not None and do_val:
#             val_set = [np.load(self.path + self.dataname + "/validation/" + val_file + ".npy") for val_file in
#                        self.val_list]
#         if do_val:
#             train_set = []
#             print("as train files num:", self.start_time - len(self.val_list))
#             for i in range(
#                     self.start_time - len(self.val_list)):  # some file in real train used be validation, left as train
#                 train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
#             train_set = np.concatenate(train_set, axis=0)
#         else:  # not do val, ie add val set ti train sets
#             train_set = []
#             print("as train files num:", self.start_time)
#             for i in range(self.start_time):
#                 train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
#             train_set = np.concatenate(train_set, axis=0)
#         return train_set, val_set, test_set
#
#     def next_online(self):
#         '''
#         :return: get the next time train and test set, conlude the first time get the init sets
#         '''
#         now_time = self.start_time + self.train_count
#         if now_time >= self.len:
#             return None, None
#         train_set = []
#         for i in range(now_time):
#             train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
#         train_set = np.concatenate(train_set, axis=0)
#         now_test = np.load(self.path + self.dataname + "/test/" + self.test_list[self.test_count] + ".npy")
#         print("will train:", self.file_list[i - 1], "will test:", self.test_list[self.test_count])
#         self.train_count += 1
#         self.test_count += 1
#         return train_set, now_test
#
#     def next_online_nopre(self):
#         '''
#         :return: get the next time train and test set, conlude the first time get the init sets
#         '''
#         now_time = self.start_time + self.train_count
#         if now_time >= (self.len - 1):
#             return None, None
#         train_set = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
#         if now_time >= (self.start_test_time - 1):
#             now_test = np.load(self.path + self.dataname + "/test/" + self.test_list[self.test_count] + ".npy")
#             print("will train:", self.file_list[now_time], "will test:", self.test_list[self.test_count])
#             self.test_count += 1
#         else:
#             now_test = None
#             print("will train:", self.file_list[now_time], "will test: None")
#         self.train_count += 1
#         return train_set, now_test
#
#     def get_time_t_data(self, time):
#         test_time = self.start_time + time
#         if test_time >= self.len:
#             print("out of time")
#             return None, None
#         print("get time:{} data".format(time))
#         print("train data numbers:", test_time)
#         print("test file:", self.test_list[time])
#         train_set = []
#         for i in range(test_time):
#             train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
#         train_set = np.concatenate(train_set, axis=0)
#         now_test = np.load(self.path + self.dataname + "/test/" + self.test_list[time] + ".npy")
#         return train_set, now_test
#
#
# class offlineDataset_withsample(Dataset):
#     """
#     data set for offline train ,and  prepare for dataloader
#     """
#
#     def __init__(self, dataset):
#         super(offlineDataset_withsample, self).__init__()
#         self.user = dataset[:, 0]
#         self.item = dataset[:, 1]
#         print("user max:", self.user.max())
#         print("user max:", self.item.max())
#         user_list = {}
#         self.item_all = np.unique(self.item)
#         for n in range(self.user.shape[0]):
#             u = self.user[n]
#             i = self.item[n]
#             try:
#                 user_list[u].append(i)
#             except:
#                 user_list[u] = [i]
#         self.user_list = user_list
#
#     def __len__(self):
#         return self.user.shape[0]
#
#     def __getitem__(self, idx):
#         user = self.user[idx]
#         item = self.item[idx]
#         # neg_item = self.item_all[0]
#         neg_item = np.random.choice(self.item_all, 1)[0]
#         while neg_item in self.user_list[user]:
#             neg_item = np.random.choice(self.item_all, 1)[0]
#         return (user, item, neg_item)


class data_prefetcher():
    def __init__(self, loader, device):  # loader是哪个
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()  # 选择给定流的上下文管理器。
        self.device_ = device
        self.preload()

    def preload(self):
        try:  # 需要生成负样本吗
            self.next_user, self.next_item, self.next_score_data = next(self.loader)
        except StopIteration:
            self.next_user = None
            self.next_item = None
            self.next_neg_item = None
            return
        with torch.cuda.stream(self.stream):
            self.next_user = self.next_user.cuda(device=self.device_, non_blocking=True)
            self.next_item = self.next_item.cuda(device=self.device_, non_blocking=True)
            self.next_neg_item = self.next_neg_item.cuda(device=self.device_, non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_user = self.next_user.long()
            self.next_item = self.next_item.long()
            self.next_neg_item = self.next_neg_item.long()

    # 加载下一stream
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        user = self.next_user
        item = self.next_item
        neg_item = self.next_neg_item
        self.preload()
        return user, item, neg_item


class testDataset(Dataset):
    '''
    Dateset subclass for test , used when the test is too large will cause out of memory!
    '''

    def __init__(self, dataset):
        super(testDataset, self).__init__()
        self.data = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class testDataset(Dataset):
    '''
    Dateset subclass for test , used when the test is too large will cause out of memory!
    '''

    def __init__(self, dataset):
        super(testDataset, self).__init__()
        self.data = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


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

class online_book_crossing(object):
    def __init__(self, stage_num=46):
        self.path = '../datasets/book-crossing'
        self.user_data, self.item_data, self.score_data = self.load()
        # self.gen_attribution(self.user_data,self.item_data)
        self.user_num, self.item_num = len(self.user_data) - 1, len(self.item_data)

        # self.u_dict, self.m_dict = self.generate(self.path, self.item_data, self.user_data)



        # self.generate_streaming(stage_num,warm_users_data)
        # self.generate_streaming(stage_num, self.score_data)
        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()

        self.split_warm_cold(self.score_data)
        # # # #
        # # # # #generate each time(stage) support and query set
        for i in tqdm(range(stage_num)):
            self.generate_current_suppAndQuery(i, types="only_new")


    # load all users and items and rating in row data
    def load(self):

        path = "../datasets/book-crossing/row"
        profile_data_path = "{}/book-crossing.user".format(path)
        score_data_path = "{}/book-crossing.inter".format(path)
        item_data_path = "{}/book-crossing.item".format(path)

        profile_data = pd.read_csv(profile_data_path, names=['user_id', 'location', 'age'],
                                   sep="\t", engine='python')
        profile_data=profile_data.drop(0)

        #用平均年龄填充
        age = profile_data[profile_data['age'].notnull()].age
        age = age.tolist()
        avg_age = int(sum([int(i) for i in age])/len(age))
        profile_data = profile_data.fillna(str(avg_age))

        item_data = pd.read_csv(
            item_data_path,
            names=['item_id', 'title', 'author', 'year', 'publisher'],
            sep="\t", engine='python', encoding="utf-8"
        )
        item_data = item_data.drop(columns=['title'])
        item_data = item_data.drop(0)

        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'item_id', 'rating'],
            sep="\t", engine='python'
        )
        score_data=score_data.drop(0)
        #!!!!!!!!!没有时间
        # score_data = score_data.sort_values(by=['timestamp'])  # 按时间戳升序
        #
        # score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        # score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data

    def gen_attribution(self, user_data, item_data):
        file_path=self.path+"/row/"
        age_list = user_data.age.unique()
        location_list = user_data.location.unique()
        self.write_attr(file_path+'b_age.txt',age_list)
        self.write_attr(file_path + 'b_location.txt', location_list)

        author_list = item_data.author.unique()
        publisher_list = item_data.publisher.unique()
        self.write_attr(file_path + 'b_author.txt', author_list)
        self.write_attr(file_path + 'b_publisher.txt', publisher_list)


    def write_attr(self,fname,data):
        f=open(fname,'w',encoding="utf-8")
        for value in data:
            f.write(str(value)+'\n')
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
        storing_path = '../datasets'
        dataset = 'movielens'
        # 按照时间划分数据集
        # sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        # start_time,  end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        # sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        # print('start time %s,  end time %s' % (start_time, end_time))
        user_warm_list, user_cold_list, user_counts = [], [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()
        n_user_ids = rat_data.user_id.nunique()

        warm, cold = pd.DataFrame(columns=['user_id', 'item_id', 'rating']), pd.DataFrame(
            columns=['user_id', 'item_id', 'rating'])
        for u_id in user_ids:
            u_info = rat_data.loc[rat_data.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id]<10 or u_count<5:continue
            if 5<self.user_current_interNum[u_id] <= 30:
                user_cold_list.append(u_id)
                self.cold_users.append(u_id)
                self.cold_users_inter = pd.concat([self.cold_users_inter,u_info]).reset_index(drop=True)

            else:
                user_warm_list.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
                if u_id in self.cold_users:
                    self.cold_users.remove(u_id)
                    warm = pd.concat([warm, self.cold_users_inter[self.cold_users_inter['user_id'] == u_id]]).reset_index(drop=True)
                    self.cold_users_inter.drop(self.cold_users_inter[self.cold_users_inter['user_id'] == u_id].index)
            user_counts.append(u_count)
        #cold = pd.concat([cold, u_info]).reset_index(drop=True)  # 取出目前为止该用户的所有交互

        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('min count: %d, avg count: %d, max count: %d' % (min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
                          'user_cold_ids': user_cold_list}
        #pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
        return user_warm_list, user_cold_list, warm, self.cold_users_inter  # pd.DataFrame(warm,columns=['user_id', 'movie_id', 'rating', 'time']),pd.DataFrame(cold,columns=['user_id', 'movie_id', 'rating', 'time'],)

    # 对warm_user的数据生成streaming data
    def generate_streaming(self, stage_num, rate_d):
        #sorted_time = rate_d.sort_values(by='time', ascending=True).reset_index(drop=True)
        stage_data_num = len(rate_d) // stage_num
        for i in tqdm(range(stage_num)):
            stage_data = rate_d.iloc[i * stage_data_num:(i + 1) * stage_data_num, :]
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
                                         names=['user_id', 'item_id', 'rating'], engine='python')
                for i in range(0, stage_id):  # 将当前stage之前的数据
                    score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                    s_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating'],
                                         engine='python')
                    score_data = pd.concat([score_data, s_data])
                    score_data.reset_index(drop=True)
                warm_users,  warm_users_data= self.split_warm_cold2(score_data)
                return warm_users, warm_users_data
            else:
                score_data_path = self.path + '/streaming/' + str(stage_id) + '.dat'
                score_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating'],
                                         engine='python')
                warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(score_data)  # cold and warm under current streaming

                # 选择在给定时间之后交互很少的用户作为cold-users（选择在当前streaming中交互很少的用户作为冷启动用户）
                return warm_users, cold_users, warm_users_data, cold_users_data
        except:
            print("read stage data wrong , may be there is no new data,finished")
            return None, None, None, None

    '''消融实验。获得full tune数据集，当下'''
    def get_current_full(self, stage_id):
        try:
            score_data = pd.read_csv(self.path + '/streaming/' + str(stage_id) + '.dat', names=['user_id', 'item_id', 'rating'],
                                     engine='python')
            users = score_data.user_id.unique()[1:]
            for i in range(0, stage_id):  # 将当前stage之前的数据
                score_data_path = self.path + '/streaming/' + str(i) + '.dat'
                s_data = pd.read_csv(score_data_path, names=['user_id', 'item_id', 'rating'],
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
        # sorted_time = rat_data.sort_values(by='time', ascending=True).reset_index(drop=True)
        # start_time,  end_time = sorted_time['time'][0], sorted_time['time'][len(rat_data) - 1]

        # sorted_users = rat_data.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)
        #print('start time %s,  end time %s' % (start_time, end_time))
        warm_users, user_counts = [], []

        new_df = pd.DataFrame()
        user_ids = rat_data.user_id.unique()[1:]
        n_user_ids = rat_data.user_id.nunique()

        warm = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
        for u_id in user_ids:
            u_info = rat_data.loc[rat_data.user_id == u_id].reset_index(drop=True)  # 是DataFrame类型的
            u_count = len(u_info)
            new_u_info = u_info.iloc[:max_count, :]
            new_df = new_df.append(new_u_info, ignore_index=True)

            self.user_current_interNum[u_id] = self.user_current_interNum.get(u_id, 0) + u_count
            if self.user_current_interNum[u_id] < 10 or u_count < 5: continue
            if u_id not in self.cold_users:
                warm_users.append(u_id)
                warm = pd.concat([warm, u_info]).reset_index(drop=True)
            user_counts.append(u_count)
        print('min count: %d, avg count: %d, max count: %d' % (min(user_counts), len(rat_data) / n_user_ids, max(user_counts)))
        return warm_users,  warm

    '''get all users and interaction and rating y based on interaction dataset score_data'''
    def get_inter_y(self, score_data):
        inters = []
        for idx, row in score_data.iterrows():
            if row['user_id'] == 'user_id':
                continue
            # if row['user_id'] not in users:
            #     users.append(row['user_id'])
            #current_all_items.add(row['movie_id'])
            inters.append([row['user_id'], row['item_id'], float(row['rating'])])

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
        #current_all_users, users_movie_inter, users_movie_inter_y = self.get_next_dataset(stage_id)
        warm_users, cold_users, warm_users_data, cold_users_data = self.get_next_dataset(stage_id, types="only_new")
        if not warm_users:
            print("***********read stage data wrong , may be there is no new data,finished***************")
            return None
        self.generate_warm_sq(stage_id,warm_users, warm_users_data, types="only_new")

        if cold_users:
            cold_inter, cold_inter_y = self.get_inter_y(cold_users_data)
            self.generate_cold_dataset(cold_users, cold_inter, cold_inter_y, stage_id)
        else:
            print("************this {} streaming has not cold users!***********".format(stage_id))
        warm_users_full, warm_users_data_full = self.get_current_full(stage_id)
        self.generate_warm_sq(stage_id, warm_users_full, warm_users_data_full, types="only_new_full")
        warm_users_f, warm_users_data_f = self.get_next_dataset(stage_id, types="not_only_new")
        self.generate_warm_sq(stage_id, warm_users_f, warm_users_data_f, types="not_only_new")
        print("**********generate dataset successfully******************")
        return True
        # 是写入文件还是直接返回suppor and query
    def generate_warm_sq(self, stage_id, warm_users, warm_users_data, types = "only_new"):
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

            if seen_movie<=5:
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

    def user_converting(self,user_row, age_list, location_list):
        age_idx = age_list.index(str(user_row.iat[2]).strip())
        location_idx = location_list.index(str(user_row.iat[1]).strip())
        return [age_idx, location_idx]

    def item_converting(self,item_row, author_list, year_list, publisher_list):
        author_idx = author_list.index(str(item_row.iat[1]).strip())
        year_idx = year_list.index(item_row.iat[2])
        publisher_idx = publisher_list.index(str(item_row.iat[3]).strip())
        # author_idx = author_list.index(item_row.iat[0, 2])1
        # year_idx = year_list.index(item_row.iat[0, 3])2
        # publisher_idx = publisher_list.index(item_row.iat[0, 4])3
        return [author_idx, year_idx, publisher_idx]

    # 将item信息转换为index格式（可以看作one-hot）
    # 用input_loading

    def load_list(self, fname):
        list_ = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():
                list_.append(line.strip())# .strip()
        return list_

    '''dataset = movielens_1m()
    item_data=dataset.item_data
    user_data=dataset.user_data
    '''

    def generate(self, master_path, item_data, user_data):
        #age_list, location_list     author_list, year_list, publisher_list
        location_list = self.load_list("{}/{}/b_location.txt".format(self.path, 'row'))
        age_list = self.load_list("{}/{}/b_age.txt".format(self.path, 'row'))

        author_list = self.load_list("{}/{}/b_author.txt".format(self.path, 'row'))
        year_list = list(item_data.year.unique())
        publisher_list = self.load_list("{}/{}/b_publisher.txt".format(self.path, 'row'))


        # dataset = movielens_1m()
        # hashmap for item information
        if not os.path.exists("{}/b_item_dict.pkl".format(master_path)):
            movie_dict = {}
            for idx, row in item_data.iterrows():
                if row['item_id'] == 'item_id': continue  # 跳过表头
                m_info = self.item_converting(row, author_list, year_list, publisher_list)
                movie_dict[row['item_id']] = m_info
            pickle.dump(movie_dict, open("{}/b_item_dict.pkl".format(master_path), "wb"))
        else:
            movie_dict = pickle.load(open("{}/b_item_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/b_user_dict.pkl".format(master_path)):
            user_dict = {}
            for idx, row in user_data.iterrows():
                if row['user_id'] == 'user_id': continue  # 跳过表头
                u_info = self.user_converting(row, age_list, location_list)
                user_dict[int(row['user_id'])] = u_info
            pickle.dump(user_dict, open("{}/b_user_dict.pkl".format(master_path), "wb"))
        else:
            user_dict = pickle.load(open("{}/b_user_dict.pkl".format(master_path), "rb"))
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
        for idx in range(len(data_file) // 4):
            s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'], engine='python')
            s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'movie_id'], engine='python')
            q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'], engine='python')
            support_x = [None, None]
            query_x = [None, None]
            support_y = None
            query_y = None
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
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

        d_t = list(zip(supp_x_s, supp_y_s, query_x_s, query_y_s))
        return d_t

    # add new users to history dataset after each roung of training
    def add_new_user_to_history(self, stage_id, types="only_new"):
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
            train_y.to_csv("{}/supp_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'], index=False)
            test_x.to_csv("{}/query_x_{}.dat".format(history_path, idx + max_idx), header=None,
                          columns=['user_id', 'movie_id'], index=False)
            test_y.to_csv("{}/query_y_{}.dat".format(history_path, idx + max_idx), header=None, columns=['rating'], index=False)
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
            if len(s_x["user_id"]) < 1: continue
            for idx, row in s_x.iterrows():
                if row['movie_id'] == "movie_id": continue
                # support_x[0].append(self.u_dict[int(row['user_id'])])
                # support_x[1].append(self.m_dict[int(row['movie_id'])])
                try:
                    support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                             0)  # 按维数0（行）拼接
                except:
                    support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
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
    # data_path = './dataset/'  # /home/ly525999/MEPM
    # data_name = 'news_3week_all'
    # file_list = ["2017010" + str(i) if i < 10 else "201701" + str(i) for i in range(1, 22)]
    # select_neg_forinteraction(path=data_path, datasetname=data_name, file_path_list=file_list)
    stage_num = 40  # 大约每五天更新一次
    states = ['warm', 'cold_user']
    # ml_dataset = online_movielens_1m()  # states,stage_num)
    ml_dataset = online_amazon()
    #ml_dataset = online_book_crossing()
    # dataset = online_yelp()

