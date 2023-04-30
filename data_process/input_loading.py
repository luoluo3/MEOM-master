
import torch.nn.functional as F
import torch

from options import config


'''
movielens
'''


class ml_item(torch.nn.Module):
    def __init__(self, config):  # 根据数据中item的content初始化，config是系统参数
        super(ml_item, self).__init__()
        self.num_rate = config['num_rate']
        self.num_genre = config['num_genre']
        self.num_director = config['num_director']
        self.num_year = config['num_year']
        self.embedding_dim = config['embedding_dim']

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        '''用于设置网络中的全连接层
        torch.nn.Linear(in_feature,out_feature,bias=True,device=None,dtype=None)
        in_feature:size of each input sample
        out_feature:size of each output sample(也是FC的神经元个数)
        bias: if set False, the layer will not learn an additive bias.Default:True
        FC's parameter shape:
          weight.shape=(out_feature,in_feature)
          bias.shape=(out_feature)
        '''
        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_year= torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim,
        )

    '''数据集中items(电影)的属性嵌入
    rate:评星
    genre:电影题材
    director:导演
    actors:演员
    '''

    # def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
    #     rate_emb = self.embedding_rate(rate_idx)
    #     genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
    #     director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
    #     actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
    #     return torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1)
    def forward(self, x):
        rate_idx, year_idx, genre_idx, director_idx = x[:,0], x[:,1], x[:,2:27], x[:,27:]
        rate_emb = self.embedding_rate(rate_idx)
        year_emb = self.embedding_year(year_idx)
        genre_emb = F.sigmoid(self.embedding_genre(genre_idx.float()))
        director_emb = F.sigmoid(self.embedding_director(director_idx.float()))
        concat_emb = torch.cat((rate_emb, year_emb, genre_emb, director_emb), 1)
        return concat_emb

class ml_user(torch.nn.Module):
    def __init__(self, config):
        super(ml_user, self).__init__()
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    '''数据集中users的属性嵌入
    gender:性别
    age:年龄
    occupation:职业
    area:国家
    '''

    # def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
    #     gender_emb = self.embedding_gender(gender_idx)
    #     age_emb = self.embedding_age(age_idx)
    #     occupation_emb = self.embedding_occupation(occupation_idx)
    #     area_emb = self.embedding_area(area_idx)
    #     return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)
    def forward(self, x):#x1是user profile
        gender_idx, age_idx, occupation_idx, area_idx = x[:,0], x[:,1], x[:,2],  x[:,3]
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)

        concat_emb = torch.cat((gender_emb, age_emb, occupation_emb,area_emb), 1)
        return concat_emb


'''
yelp
'''


class yelp_item(torch.nn.Module):
    # def __init__(self, config):#根据数据中item的content初始化，config是系统参数
    #     super(yelp_item_embedding, self).__init__()
    def __init__(self, config):
        super(yelp_item, self).__init__()
        self.n_city = config['num_city']
        self.n_state = config['num_state']
        self.n_postal_code = config['num_postal_code']
        self.n_stars = config['num_item_stars']
        self.n_count = config['num_item_review_count']
        self.n_cate = config['num_categories']
        self.embedding_dim = config['embedding_dim']

        self.embedding_city = torch.nn.Embedding(
            num_embeddings=self.n_city,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_state = torch.nn.Embedding(
            num_embeddings=self.n_state,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_code = torch.nn.Embedding(
            num_embeddings=self.n_postal_code,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.n_stars,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_count = torch.nn.Embedding(
            num_embeddings=self.n_count,
            embedding_dim=self.embedding_dim,
        )

        self.embedding_cate = torch.nn.Linear(
            in_features=self.n_cate,
            out_features=self.embedding_dim,
            bias=False
        )


    def forward(self,x):
        city_idx, state_idx, code_idx, stars_idx , count_idx, cate_idx  = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5:]
        city_emb = self.embedding_city(city_idx)
        state_emb = self.embedding_state(state_idx)
        code_emb = self.embedding_code(code_idx)
        stars_emb = self.embedding_stars(stars_idx)
        count_emb = self.embedding_count(count_idx)
        cate_emb = F.sigmoid(self.embedding_cate(cate_idx.float()))

        concat_emb = torch.cat((city_emb, state_emb, code_emb, stars_emb,count_emb,cate_emb), 1)
        return concat_emb


class yelp_user(torch.nn.Module):
    def __init__(self, config):
        super(yelp_user, self).__init__()
        self.n_count = config['num_count']

        self.n_fans = config['num_fans']
        self.n_stars = config['num_stars']

        self.n_c_hot = config['num_c_hot']
        self.n_c_more = config['num_c_more']
        self.n_c_profile = config['num_c_profile']
        self.n_c_cute = config['num_c_cute']
        self.n_c_list = config['num_c_list']

        self.n_c_writer = config['num_c_writer']
        self.n_c_photos = config['num_c_photos']

        self.embedding_dim = config['embedding_dim']


        self.embedding_writer = torch.nn.Embedding(
            num_embeddings=self.n_c_writer,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_photos = torch.nn.Embedding(
            num_embeddings=self.n_c_photos,
            embedding_dim=self.embedding_dim,
        )

        self.embedding_count = torch.nn.Embedding(
            num_embeddings=self.n_count,
            embedding_dim=self.embedding_dim,
        )

        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.n_fans,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.n_stars,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_hot = torch.nn.Embedding(
            num_embeddings=self.n_c_hot,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_more = torch.nn.Embedding(
            num_embeddings=self.n_c_more,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_profile = torch.nn.Embedding(
            num_embeddings=self.n_c_profile,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_cute = torch.nn.Embedding(
            num_embeddings=self.n_c_cute,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_list = torch.nn.Embedding(
            num_embeddings=self.n_c_list,
            embedding_dim=self.embedding_dim,
        )



    def forward(self,x):
        count, fans, stars, c_hot, c_more, c_profile, c_cute, c_list, c_writer, c_photos= x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5],x[:, 6],x[:, 7], x[:, 8], x[:, 9]
        count_emb = self.embedding_count(count)
        fans_emb = self.embedding_fans(fans)
        stars_emb = self.embedding_stars(stars)
        c_hot_emb = self.embedding_hot(c_hot)
        c_more_emb = self.embedding_more(c_more)
        c_profile_emb = self.embedding_profile(c_profile)
        c_cute_emb = self.embedding_cute(c_cute)
        c_list_emb = self.embedding_list(c_list)
        c_writer_emb = self.embedding_writer(c_writer)
        c_photos_emb = self.embedding_photos(c_photos)

        concat_emb = torch.cat((count_emb, fans_emb, stars_emb,c_hot_emb, c_more_emb,c_profile_emb,c_cute_emb,c_list_emb,c_writer_emb,c_photos_emb), 1)
        return concat_emb


'''
Amazon
'''

class amazon_item(torch.nn.Module):
    def __init__(self, config):  # 根据数据中item的content初始化，config是系统参数
        super(amazon_item, self).__init__()

        self.num_cate = config['num_cate']
        self.num_title = config['num_title']
        self.num_price = config['num_price']
        self.num_brand = config['num_brand']
        # self.num_type = config['num_type']
        # self.num_rank = config['num_rank']
        self.embedding_dim = config['embedding_dim']

        #多个的用Linner 单个用Embedding
        self.embedding_cate = torch.nn.Linear(
            in_features=self.num_cate,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_title = torch.nn.Embedding(
            num_embeddings=self.num_title,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_price = torch.nn.Embedding(
            num_embeddings=self.num_price,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_brand = torch.nn.Embedding(
            num_embeddings=self.num_brand,
            embedding_dim=self.embedding_dim,
        )


    def forward(self, x):
        title_idx, price_idx, brand_idx, cate_idx = x[:, 0], x[:, 1], x[:, 2], x[:, 3:]
        cate_emb = F.sigmoid(self.embedding_cate(cate_idx.float()))
        title_emb = self.embedding_title(title_idx)
        price_emb = self.embedding_price(price_idx)
        brand_emb = self.embedding_brand(brand_idx)
        concat_emb = torch.cat((cate_emb, title_emb, price_emb, brand_emb), 1)
        return concat_emb


class BKUserLoading(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BKUserLoading, self).__init__()
        self.age_dim = config['n_age_bk']
        self.location_dim = config['n_location']
        self.embedding_dim = embedding_dim

        self.emb_age = torch.nn.Embedding(num_embeddings=self.age_dim, embedding_dim=self.embedding_dim)
        self.emb_location = torch.nn.Embedding(num_embeddings=self.location_dim, embedding_dim=self.embedding_dim)

    def forward(self, x1):
        age_idx, location_idx = x1[:,0], x1[:,1]
        age_emb = self.emb_age(age_idx)
        location_emb = self.emb_location(location_idx)
        concat_emb = torch.cat((age_emb, location_emb), 1)
        return concat_emb

class BKItemLoading(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BKItemLoading, self).__init__()
        self.year_dim = config['n_year_bk']
        self.author_dim = config['n_author']
        self.publisher_dim = config['n_publisher']
        self.embedding_dim = embedding_dim

        self.emb_year = torch.nn.Embedding(num_embeddings=self.year_dim, embedding_dim=self.embedding_dim)
        self.emb_author = torch.nn.Embedding(num_embeddings=self.author_dim, embedding_dim=self.embedding_dim)
        self.emb_publisher = torch.nn.Embedding(num_embeddings=self.publisher_dim, embedding_dim=self.embedding_dim)

    def forward(self, x2):#x2是item的profile
        author_idx, year_idx, publisher_idx = x2[:,0], x2[:,1], x2[:,2]
        year_emb = self.emb_year(year_idx)
        author_emb = self.emb_author(author_idx)
        publisher_emb = self.emb_publisher(publisher_idx)
        concat_emb = torch.cat((year_emb, author_emb, publisher_emb), 1)
        return concat_emb

