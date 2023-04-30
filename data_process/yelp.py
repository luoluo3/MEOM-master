import collections
import datetime
import os
import pickle
import re
from math import floor
import random

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

class online_yelp(object):
    def __init__(self, stage_num=38):
        self.path = '../datasets/yelp'
        self.user_data, self.item_data, self.score_data = self.load()
        # self.gen_attribution(self.item_data)

        # self.score_data = self.filter_data(self.score_data)

        print(len(self.score_data), self.score_data.user_id.nunique(), self.score_data.item_id.nunique())
        # user_ids = list(set(self.score_data.user_id))
        self.user_num, self.item_num = self.user_data.user_id.nunique(), self.item_data.item_id.nunique()

        # self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num, self.score_data)

        self.u_dict, self.i_dict = self.generate(self.path, self.item_data, self.user_data)
        # warm_users, cold_users, warm_users_data, cold_users_data = self.split_warm_cold(self.score_data)

        # self.generate_streaming(stage_num,warm_users_data)


        self.user_current_interNum = {}  # 记录到当前stage为止出现的所有用户及其交互数
        self.cold_users = []
        self.cold_users_inter = pd.DataFrame()
        # # # # #generate each time(stage) support and query set
        for i in tqdm(range(stage_num)):
            self.generate_current_suppAndQuery(i, types="only_new")

    # load all users and items and rating in row data
    def load(self):
        path = "../datasets/yelp/row"
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
        # if not warm_users:
        #     print("***********read stage data wrong , may be there is no new data,finished***************")
        #     return None
        # self.generate_warm_sq(stage_id, warm_users, warm_users_data, types="only_new")
        #
        # if cold_users:
        #     cold_inter, cold_inter_y = self.get_inter_y(cold_users_data)
        #     self.generate_cold_dataset(cold_users, cold_inter, cold_inter_y, stage_id)
        # else:
        #     print("************this {} streaming has not cold users!***********".format(stage_id))
        # # warm_users_full, warm_users_data_full = self.get_current_full(stage_id)
        # # self.generate_warm_sq(stage_id, warm_users_full, warm_users_data_full, types="only_new_full")

        if stage_id>8:
            self.user_current_interNum = {}
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
            idx = cate_list.index(re.sub(r'\'', '', cate))           #re.sub(r'\([^()]*\)', '', cate)
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
    '''['city', 'state', 'postal_code',  'item_stars','item_review_count', 'is_open', 'categories']

        ['user_review_count','fans', 'average_stars', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
         'compliment_list', 
         'compliment_writer', 'compliment_photos']'''

    def generate(self, master_path, item_data, user_data):
        # item
        city_list = self.load_list("{}/{}/city.txt".format(self.path, 'row'))
        state_list = self.load_list("{}/{}/state.txt".format(self.path, 'row'))
        post_list = self.load_list("{}/{}/postal.txt".format(self.path, 'row'))
        cate_list = self.load_list("{}/{}/cate.txt".format(self.path, 'row'))
        i_stars_list = list(item_data.item_stars.unique())
        i_count_list = list(item_data.item_review_count.unique())
        is_open_list = list(item_data.is_open.unique())

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

        # hashmap for item information
        if not os.path.exists("{}/item_dict.pkl".format(master_path)):
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
    '''obtain next full dataset from previous each stage new dataset'''

    def next_full_data(self, stage_id, path):
        # 读取<=stage_id之前的所有stage_dataset_only_new里的数据，然后如果support_x里面的user相同的话就合并
        users= collections.OrderedDict()
        for stage in range(stage_id+1):
            dataset_path = path + "/stage_dataset_only_new/" + str(stage_id)
            try:  # get all filename
                data_file = os.listdir(dataset_path)
            except:
                print("filename error or filepath error!")
            supp_x_s ,supp_y_s = [],[]
            query_x_s ,query_y_s = [],[]
            for idx in range(len(data_file) // 4):
                s_x = pd.read_csv('{}/supp_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],header=None,
                                  engine='python')
                s_y = pd.read_csv('{}/supp_y_{}.dat'.format(dataset_path, idx), names=['rating'], header=None,engine='python')
                q_x = pd.read_csv('{}/query_x_{}.dat'.format(dataset_path, idx), names=['user_id', 'item_id'],header=None,
                                  engine='python')
                q_y = pd.read_csv('{}/query_y_{}.dat'.format(dataset_path, idx), names=['rating'],header=None, engine='python')
                support_x = [None, None]
                query_x = [None, None]
                support_y = None
                query_y = None
                if len(s_x["user_id"]) < 1: continue
                for idx, row in s_x.iterrows():
                    if row['item_id'] == "item_id": continue
                    try:
                        support_x[0] = torch.cat((support_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                                 0)  # 按维数0（行）拼接
                    except:
                        support_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
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
                        query_x[0] = torch.cat((query_x[0], torch.tensor([self.u_dict[int(row['user_id'])]])),
                                               0)  # 按维数0（行）拼接
                    except:
                        query_x[0] = torch.tensor([self.u_dict[int(row['user_id'])]])
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

                if s_x.user_id.unique() in users.keys():
                    # filter_score = pd.concat([filter_score, score.loc[score.user_id == user]])
                    # users[s_x.user_id.unique()][0] = pd.concat([users[s_x.user_id.unique()][0], s_x])
                    # users[s_x.user_id.unique()][1] = pd.concat([users[s_x.user_id.unique()][1], s_y])
                    # users[s_x.user_id.unique()][2] = pd.concat([users[s_x.user_id.unique()][2], q_x])
                    # users[s_x.user_id.unique()][3] = pd.concat([users[s_x.user_id.unique()][3], q_y])
                    users[s_x.user_id.unique()][0][0] = torch.cat((users[s_x.user_id.unique()][0][0],support_x[0]),0)
                    users[s_x.user_id.unique()][0][1] = torch.cat((users[s_x.user_id.unique()][0][1], support_x[1]), 0)
                    users[s_x.user_id.unique()][1] = torch.cat((users[s_x.user_id.unique()][1], support_y), 0)
                    users[s_x.user_id.unique()][2][0] = torch.cat((users[s_x.user_id.unique()][2][0], query_x[0]), 0)
                    users[s_x.user_id.unique()][2][1] = torch.cat((users[s_x.user_id.unique()][2][1], query_x[1]), 0)
                    users[s_x.user_id.unique()][3] = torch.cat((users[s_x.user_id.unique()][3], query_y), 0)
                else:
                    users[s_x.user_id.unique()]=[support_x,support_y,query_x,query_y]
        return list(users.values())


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


if __name__ == "__main__":
    # data_path = './dataset/'  # /home/ly525999/MEPM
    # data_name = 'news_3week_all'
    # file_list = ["2017010" + str(i) if i < 10 else "201701" + str(i) for i in range(1, 22)]
    # select_neg_forinteraction(path=data_path, datasetname=data_name, file_path_list=file_list)
    stage_num = 38  # 大约每五天更新一次
    states = ['warm', 'cold_user']
    # ml_dataset = online_movielens_1m()  # states,stage_num)
    # ml_dataset = online_amazon()
    #ml_dataset = online_book_crossing()
    dataset = online_yelp()
