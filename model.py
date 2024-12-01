from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
import random
import time
import torch.nn.init as init
from utils import *
from dgl.nn import GraphConv
import dgl
SEED = 123

# PyTorch
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def init_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name or 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)




class ourmodel(nn.Module):


    def __init__(self, g,feature,h1, neigbor_node,num_aspects, num_users,num_items,dropout_rate=0.5,batch_meta_g =None):
        super(ourmodel,self).__init__()  
        self.num_users = num_users
        self.num_items = num_items
        self.num_aspects = num_aspects
        self.g= g
        self.feature= feature
        self.h1 = h1
        self.dropout_rate = dropout_rate
        self.aspect_emb= nn.Embedding(50000,self.num_aspects*self.h1)
        self.feature_fusion= nn.Linear(self.num_aspects*self.h1,self.num_aspects*self.h1,bias=True)
        self.attn = nn.Linear(4*self.num_aspects*self.h1,self.num_aspects*self.h1,bias=True)
        self.lstm = nn.LSTM(input_size= self.num_aspects*self.h1, hidden_size= self.num_aspects*self.h1, batch_first=True)
        self.GCN = GCN(self.num_aspects*self.h1, self.num_aspects,dropout_rate)
        self.ANR_RatingPred = ANR_RatingPred(self.num_aspects,self.num_users,self.num_items,dropout_rate)
        self.Aspect_Importance = Aspect_Importance(self.num_aspects)
        self.aspect_linear= nn.Linear(100,self.num_aspects*self.h1)

        self._init_weights()

        self.softmax = nn.Softmax(dim=-2)

        self.leaky = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.linear_layer = nn.Linear(self.num_aspects*h1*2, self.num_aspects)
        self.node_agg_layer = nn.Linear(self.num_aspects*self.h1, self.num_aspects)
        self.edge_agg_layer = nn.Linear(self.num_aspects*self.h1, self.num_aspects)

        self.batch_meta_g = batch_meta_g ## 새로 추가
        self.SUBGCN = SUBGCN(self.num_aspects*self.h1, self.num_aspects*self.h1) ## 새로 추가
        self.attention_linear = nn.Linear(self.num_aspects*self.h1, 1) ## 새로 추가
        self.bn1 = nn.BatchNorm1d(self.num_aspects*self.h1)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_uniform_(param.data, nonlinearity='leaky_relu')
                    elif 'weight_hh' in name:
                        nn.init.kaiming_uniform_(param.data, nonlinearity='leaky_relu')
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
        

    def forward(self, path_tensor,uid, iid, nid):
        batch, n_node,  depth = path_tensor.size()

        self.node_emb = self.feature[path_tensor]  # b x n x d x 100
        self.node_emb = self.aspect_linear(self.node_emb) 
        self.node_emb = self.node_emb.view(batch*n_node,depth,-1)
        
        self.node_aspect = self.aspect_emb(path_tensor)  # b x n x d x K x h
        self.node_aspect = F.layer_norm(self.node_aspect, [self.node_aspect.size(-1)])

        self.node_aspect_emb = self.node_aspect.view(batch*n_node,depth,self.num_aspects*self.h1)

        if self.batch_meta_g != None:        
            # b*n x depth x K
            self.center_path_aspect_emb= self.tanh(self.lstm(self.node_aspect_emb[:,1:,:])[0])
            self.copied_center_path_aspect_emb = self.center_path_aspect_emb.clone()

            # b*n*depth x K
            self.center_lstm_node_feature = self.copied_center_path_aspect_emb.reshape(batch*n_node*(depth-1), self.num_aspects*self.h1)
            self.center_sub_gcn_output = self.SUBGCN(self.batch_meta_g, self.center_lstm_node_feature)
            
            # b*n x depth x K
            self.center_path_attention_weight = F.softmax(F.tanh(self.attention_linear(self.center_sub_gcn_output.view(batch*n_node, (depth-1), self.num_aspects*self.h1))),dim=1)
            self.center_path_aspect_emb = (self.center_path_attention_weight*self.center_path_aspect_emb).sum(dim=1).unsqueeze(1)

            # b*n x depth x K
            self.filp_node_aspect_emb = torch.flip(self.node_aspect_emb, dims=[1])
            self.neighbor_path_aspect_emb = self.tanh(self.lstm(self.filp_node_aspect_emb[:,1:,:])[0])
            self.copied_neighbor_path_aspect_emb = self.neighbor_path_aspect_emb.clone()

            # b*n*depth x K
            self.neighbor_lstm_node_feature = self.copied_neighbor_path_aspect_emb.reshape(batch*n_node*(depth-1), self.num_aspects*self.h1)
            self.neighbor_sub_gcn_output = self.SUBGCN(self.batch_meta_g, self.neighbor_lstm_node_feature)
            
            # b*n x depth x K
            self.neighbor_path_attention_weight = F.softmax(F.tanh(self.attention_linear(self.neighbor_sub_gcn_output.view(batch*n_node, (depth-1), self.num_aspects*self.h1))),dim=1)
            self.neighbor_path_aspect_emb = (self.neighbor_path_attention_weight*self.neighbor_path_aspect_emb).sum(dim=1).unsqueeze(1)
        else:
            # b*n x 1 x K
            self.center_path_aspect_emb= self.leaky(self.bn1(self.lstm(self.node_aspect_emb[:,1:,:])[0][:,-1])).unsqueeze(1)
            filp_node_aspect_emb = torch.flip(self.node_aspect_emb, dims=[1])
            self.neighbor_path_aspect_emb = self.leaky(self.bn1(self.lstm(filp_node_aspect_emb[:,1:,:])[0][:,-1])).unsqueeze(1)
       
       
        path_aspect_emb = torch.cat([self.center_path_aspect_emb,self.neighbor_path_aspect_emb],dim=1) # b*n x 2 x K\
        sd_node_asepct_emb = self.node_emb[:,[0,-1]].reshape(batch*n_node,-1,self.num_aspects*self.h1) # b*n x 2 x K

        self.fusion_feature = self.feature_fusion(path_aspect_emb + sd_node_asepct_emb)  # b*n x 2 x K
        fusion_feature = F.dropout(self.fusion_feature, p=self.dropout_rate, training=self.training)
        fusion_feature = F.layer_norm(fusion_feature, [fusion_feature.size(-1)])
        fusion_feature = self.leaky(fusion_feature)

        self.edge_weight = self.attn(torch.cat([fusion_feature, sd_node_asepct_emb], dim=1).reshape(batch*n_node,-1))
        self.edge_weight = F.dropout((self.edge_weight), p=self.dropout_rate, training=self.training)        
        self.edge_weight = F.layer_norm(self.edge_weight, [self.edge_weight.size(-1)])
        self.edge_weight= self.tanh(self.edge_weight.reshape(batch*n_node,-1))

        node_path_pair = path_pair(path_tensor)
        update_edge_weights(self.g,node_path_pair, self.edge_weight) 
        self.node_emb = self.node_emb.view(batch,n_node,depth,-1)
        h = self.GCN(self.g, self.node_aspect[:, 0, 0])

        self.user_h, self.item_h = h[:self.num_users],h[self.num_users:]


        node_user = self.user_h[uid].squeeze()
        node_item = self.item_h[iid].squeeze()
        node_n_item = self.item_h[nid].squeeze()

 
     
        self.pos_cos = F.pairwise_distance(node_user, node_item,1)
        self.neg_cos =torch.mean(torch.stack([F.pairwise_distance(node_user, node_n_item[:,i].squeeze(), 1) for i in range(nid.long().shape[1])],dim=1),dim=1)
        alpha = 0.5
        positive_loss = torch.mean(self.pos_cos)
        negative_loss = torch.mean(torch.clamp(1 - self.neg_cos, min=0.0))
        triple_loss = alpha * positive_loss + (1 - alpha) * negative_loss

        self.user_attn, self.item_attn = self.Aspect_Importance(self.user_h, self.item_h)
        self.rating_pred,loss= self.ANR_RatingPred(self.user_h, self.item_h, self.user_attn, self.item_attn,uid, iid, nid)
  
        return self.rating_pred,loss, triple_loss

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, in_dim, weight=False)
        self.bn1 = nn.BatchNorm1d(in_dim) 
        self.gcn2 = GraphConv(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.BatchNorm1d(in_dim) 
        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)  # 드롭아웃 추가
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(in_dim,out_dim)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, GraphConv) and m.weight is not None:
                init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, graph, feature):
        edge_weight = graph.edata['weight']

        h = self.gcn1(graph, feature, edge_weight=edge_weight)
        h = self.bn1(self.dropout(h))
        self.h1_layer = self.activate(h)
        self.h2_layer = self.gcn2(graph, self.h1_layer)
        h = self.bn2(self.h2_layer)
        #self.h2_layer = self.linear(h)
        h = self.softmax(h)

        return h


   
def init_variable(dim1, dim2, name=None):
    return nn.Parameter(torch.randn(dim1, dim2))

class Aspect_Importance(nn.Module):

    def __init__(self,  num_aspect):

        super(Aspect_Importance, self).__init__()
        self.W_a = nn.Parameter(torch.Tensor(num_aspect*1, num_aspect*1), requires_grad = True)
        self.W_u = nn.Parameter(torch.Tensor(num_aspect*1, num_aspect*1), requires_grad = True)
        self.w_hu = nn.Parameter(torch.Tensor(num_aspect*1, num_aspect*1), requires_grad = True)

        # Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_i = nn.Parameter(torch.Tensor(num_aspect*1, num_aspect*1), requires_grad = True)
        self.w_hi = nn.Parameter(torch.Tensor(num_aspect*1, num_aspect*1), requires_grad = True)


        self.affinityMatrix = None

        self._init_weights()

        self.activate = nn.ReLU()
        self.softmax = nn.Softmax()

        self.bn1 = nn.BatchNorm1d(num_aspect*1) 
        self.bn2 = nn.BatchNorm1d(num_aspect*1) 
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_a, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_u, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_hu, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_i, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_hi, nonlinearity='relu')
    def forward(self, userAspRep, itemAspRep, verbose = 0):

        affinityMatrix = torch.matmul(userAspRep, self.W_a)  # u x a
        self.affinityMatrix = torch.matmul(affinityMatrix, itemAspRep.T) # u x i 
        # Non-Linearity: ReLU
        #self.affinityMatrix = self.softmax(self.affinityMatrix)
        H_u_1 = self.bn1(torch.matmul(userAspRep, self.W_u )) # u x a
        H_u_2 = self.bn2(torch.matmul(itemAspRep, self.W_i)) # i x a
        H_u_2 = torch.matmul(self.affinityMatrix, H_u_2)# u x a
        H_u = self.activate(H_u_1 + H_u_2)


        # User Aspect-level Importance
        userAspImpt = torch.matmul(H_u, self.w_hu)
        userAspImpt = F.softmax(userAspImpt, dim = 1)

        H_i_1 = self.bn2(torch.matmul(itemAspRep, self.W_i))
        H_i_2 = self.bn1(torch.matmul(userAspRep, self.W_u))
        H_i_2 = torch.matmul(self.affinityMatrix.T,H_i_2)
        H_i = self.activate(H_i_1 + H_i_2)
  
        itemAspImpt = torch.matmul(H_i, self.w_hi)
        itemAspImpt = F.softmax(itemAspImpt, dim = 1)

        return userAspImpt, itemAspImpt
class ANR_RatingPred(nn.Module):

    def __init__(self, num_aspect, num_users, num_items,dropout_rate):
        super(ANR_RatingPred, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_aspect =num_aspect
        self.userAspRepDropout = nn.Dropout(p = dropout_rate)
        self.itemAspRepDropout = nn.Dropout(p = dropout_rate)

        self.item_bias = nn.Parameter(torch.zeros(self.num_items), requires_grad = True)
 
        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = nn.Parameter(torch.zeros(self.num_users, 1))
        self.uid_userOffset.requires_grad = True

        self.iid_itemOffset = nn.Parameter(torch.zeros(self.num_items, 1))
        self.iid_itemOffset.requires_grad = True

        # Initialize Global Bias with 0
        self.item_bias.data.fill_(0)
        self.uid_userOffset.data.fill_(0)
        self.iid_itemOffset.data.fill_(0)

        self.user_latent = init_variable(num_users, 100)
        self.item_latent = init_variable(num_items, 100)

        nn.init.normal_(self.user_latent, mean=0, std=0.01)
        nn.init.normal_(self.item_latent, mean=0, std=0.01)

        self.alpha1 = nn.Parameter(torch.tensor(1.))
        self.xuij,self.rate_matrix_i,self.rate_matrix_j  = None,None,None

    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt, user_b,item_b,n_item_b,verbose = 0):

        userAspRep = self.userAspRepDropout(userAspRep) # u x a
        itemAspRep = self.itemAspRepDropout(itemAspRep) # i x a


        u_factors = self.user_latent[user_b]
        i_factors = self.item_latent[item_b]
        j_factors = self.item_latent[n_item_b]
        

        rate_matrix1_i = torch.sum(u_factors * i_factors, dim=2)# b 
        rate_matrix1_j = torch.sum(u_factors * j_factors, dim=2)
        rate_matrix1 = torch.matmul(self.user_latent, self.item_latent.t()) # u x i


        userAspRep=  userAspRep  + self.uid_userOffset
        itemAspRep = itemAspRep  + self.iid_itemOffset

        u_factors = userAspRep[user_b]
        i_factors = itemAspRep[item_b]
        j_factors = itemAspRep[n_item_b]
        
        #u_factors = self.user_linear(u_factors)
        rate_matrix2_i = torch.sum(u_factors * i_factors, dim=2)# b 
        rate_matrix2_j = torch.sum(u_factors * j_factors, dim=2) 
        rate_matrix2 = torch.matmul(userAspRep, itemAspRep.t())
       
        self.rate_matrix_i = rate_matrix1_i + self.alpha1*rate_matrix2_i + self.item_bias[item_b]
        self.rate_matrix_j = rate_matrix1_j + self.alpha1*rate_matrix2_j + self.item_bias[n_item_b].squeeze()

        self.xuij = self.rate_matrix_i - self.rate_matrix_j
        rate_matrix = rate_matrix1 + self.alpha1*rate_matrix2 + self.item_bias.T
        return rate_matrix, -torch.mean(torch.log(torch.clamp(F.sigmoid(self.xuij), min=1e-10, max=1.0)))






class SUBGCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super(SUBGCN, self).__init__()
        self.gcn1 = GraphConv(in_dim, in_dim, weight=False, norm='both')
        self.bn1 = nn.BatchNorm1d(in_dim) 
        self.gcn2 = GraphConv(in_dim, in_dim, weight=False, norm='both')
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.gcn3 = GraphConv(in_dim, in_dim, weight=False, norm='both')
        self.bn3 = nn.BatchNorm1d(in_dim) 
        self.activate = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)  # 드롭아웃 추가
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, graph, feature):
        # edge_weight = graph.edata['weight']

        h = self.gcn1(graph, feature)
        h = self.bn3(h)
        h = self.activate(h)
        h = self.gcn2(graph, h)

        return h