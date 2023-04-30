import torch
from options import *
class BaseModel(torch.nn.Module):
    def __init__(self, alpha, eta, rec_module):
        super(BaseModel, self).__init__()

        #将这两个参数加入BaseModel模型的参数中，作为可训练对象
        self.alpha = torch.nn.Parameter(torch.FloatTensor(alpha), requires_grad=True)
        self.eta = torch.nn.Parameter(torch.FloatTensor(eta), requires_grad=True)

        self.rec_model = rec_module#推荐模型

    def forward(self, x):
        _, _, rec_value = self.rec_model(x)
        return rec_value

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


