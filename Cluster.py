# 高斯混合模型
from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
import torch
import random
import copy
from collections import OrderedDict

'''
clustering based on user_dict 
'''
class task_cluster():
    def __init__(self, data, k):
        self.min_a, self.max_a = [], []
        attributions=self.data_map(list(data.values()),10)

        self.data = torch.tensor(attributions).float()  #torch.tensor(list(data.values())).float()
        # self.data = data
        self.task_id = list(data.keys())
        # self.tasks_dynamic_prefer = tasks_dynamic_prefer
        self.k = k
        self.task_n = len(data.keys())


        self.labels, self.center = self.forward()


    def data_map(self,data, n):
        #对数据进行归一化处理，每个属性都映射到0-1之间
        new_data=[]
        for i in range(n): #性别属性只有两个取值0，1
            attribution = [data[j][i] for j in range(len(data))]
            max_a,min_a= max(attribution),min(attribution)
            self.min_a.append(min_a)
            self.max_a.append(max_a)
            new_data.append([(num-min_a)/(max_a-min_a) for num in attribution])
        a1 = [data[j][0] for j in range(len(data))]
        return list(zip(a1,new_data[0],new_data[1],new_data[2]))

    def user_map(self,data):
        a0 = [data[j][0] for j in range(len(data))]
        a1 = [(data[j][1]-self.min_a[1])/(self.max_a[1]-self.min_a[1]) for j in range(len(data))]
        a2 = [(data[j][2] - self.min_a[2])/(self.max_a[2]-self.min_a[2]) for j in range(len(data))]
        a3 = [(data[j][3] - self.min_a[3])/(self.max_a[3]-self.min_a[3]) for j in range(len(data))]
        return list(zip(a0,a1,a2,a3))


    def distance(self, p1, p2):
        return torch.sum((p1-p2).float()**2).sqrt()# torch.sum((torch.tensor(p1)-torch.tensor(p2))**2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        # n = self.data.size(0)
        rand_id = random.sample(range(self.task_n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])  #根据随机生成的id相当于随机选择样本作为类中心
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        set1 = set(old_center)
        set2 = set(new_center)
        return set1 == set2

    # 先基于profile对所有用户聚类
    def forward(self):
        center = self.generate_center()
        # n = self.data.size(0)
        # labels = torch.zeros(self.task_n).long()
        labels = OrderedDict()
        flag = False
        numToCluster = 30
        while not flag and numToCluster:
            old_center = copy.deepcopy(center)
            for i in range(self.task_n):
                cur = self.data[i]
                min_dis = float('inf')
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                c_d = [id for id, c in labels.items() if c == j]
                if len(c_d)==0:
                    center[j] = torch.zeros_like(self.data[0])
                else:
                    center[j] = torch.mean(torch.stack([self.data[i] for i in c_d],dim=0),dim=0)

            flag = self.converge(old_center, center)
            numToCluster-=1

        for j in range(self.k):
            c_d = [id for id, c in labels.items() if c == j]
            if len(c_d) == 0:
                center[j] = torch.zeros_like(self.data[0])
            else:
                center[j] = torch.mean(torch.stack([self.data[i] for i in c_d], dim=0), dim=0)
        # self.show(self.data,list(labels.values()),[i for i in range(self.k)])

        # self.dynamic_center = self.get_dynamic_prefer(labels)
        return labels, center

    def update_cluster(self, data):
        '''定期更新类中心表示'''
        self.data = torch.tensor(list(data.values()))
        self.task_id = list(data.keys())
        self.task_n = len(data.keys())
        center = self.center
        # labels = torch.zeros(self.task_n).long()
        labels = OrderedDict()
        flag = False
        while not flag:
            old_center = copy.deepcopy(center)

            #for i in range(self.task_n):
            for i in range(self.task_id):
                cur = self.data[i]
                min_dis = 10 * 9
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                center[j] = torch.mean(self.data[labels == j], dim=0)

            flag = self.converge(old_center, center)
        self.center = center
        self.labels = labels
        return labels, center

    def get_dynamic_center(self, dynamic_data):
        '''定期更新类中心表示
        万一dynamic_data[i]没有
        '''
        dynamic_center = []
        # labels = torch.tensor(self.labels)
        shape_dp = list(dynamic_data.values())[0]
        for i in range(self.k): # get each cluster task's task_dynamic_prefer to get mean
            # idx = labels.nonzero(labels == i).squeeze().tolist()
            idx = [id for id in self.labels if self.labels[id]==i]
            all_d = [dynamic_data[id] for id in idx if dynamic_data.get(id)!=None]
            if all_d==[]:
                center = torch.zeros_like(shape_dp)
                dynamic_center.append(center)
            else:
                center = torch.stack(all_d)  # tasks_dynamic_prefer[id].float()
                dynamic_center.append(torch.mean(center, dim=0))
            # d_c = torch.mean(center, dim=0)
            # if len(d_c.size())==0:
            #     d_c = torch.zeros_like(shape_dp)
            # dynamic_center.append(torch.mean(center, dim=0))

        return dynamic_center

    def get_dynamic_prefer(self, c_user_profile, dynamic_center):  # task_dynamic_data):
        # 先计算soft-cluser所属类，然后加权求d_p，基于距离定义与各类的权重
        cus_dynamic_pre=OrderedDict()
        c_profiles = self.user_map(list(c_user_profile.values()))
        # for c_id, c_profile in c_user_profile.items():
        c_ids = list(c_user_profile.keys())
        for id in range(len(c_ids)):
            c_id = c_ids[id]
            c_profile = c_profiles[id]
            dis=[]
            for c in self.center:
                dis.append(self.distance(c, torch.tensor(c_profile).float()))

            # weight = torch.nn.functional.softmax(torch.stack(dis), dim=0)  # 有可能是数值太大了
            weight = torch.stack(dis)/torch.sum(torch.stack(dis)).item()
            # dynamic_center = self.get_dynamic_center(task_dynamic_data)
            d_att = [dynamic_center[i]*weight[i].item() for i in range(self.k)]
            c_user_dynamic_pre = torch.mean(torch.stack(d_att),dim=0)
            cus_dynamic_pre[c_id] = c_user_dynamic_pre# cus_dynamic_pre.append(c_user_dynamic_pre)  #cus_dynamic_pre[c_id] = c_user_dynamic_pre
        return cus_dynamic_pre

    def show(self,data,ys,clusters):
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        ys = np.array(ys)
        data = np.array([[data[i][0]+data[i][1],data[i][2]+data[i][3]]for i in range(len(data))])
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(ys == cluster)[:50]
            # row_ix = [i for i in range(len(ys)) if ys[i]==cluster]
            # 创建这些样本的散布

            pyplot.scatter(data[row_ix, 0], data[row_ix, 1])

        # 绘制散点图
        pyplot.show()



# class Task_Cluster():
#     def __init__(self,k):
#         self.gmm=GaussianMixture(n_components=2)
#
#
#
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# # 定义模型
# model = GaussianMixture(n_components=2)
# # 模型拟合
# model.fit(X)
# # 为每个示例分配一个集群
# yhat = model.predict(X)
# # 检索唯一群集
# clusters = unique(yhat)
# # 为每个群集的样本创建散点图
# for cluster in clusters:
# # 获取此群集的示例的行索引
#     row_ix = where(yhat == cluster)
# # 创建这些样本的散布
#     pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # 绘制散点图
# pyplot.show()
#
#
# '''get a task initial parameter based on membership of each cluster'''
# def get_initail_parms(membership,basemdoels):
#     pass
