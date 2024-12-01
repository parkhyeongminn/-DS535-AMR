# hetero
import torch
from scipy.sparse import csr_matrix
import dgl
from tqdm import tqdm
from scipy import sparse
import torch.optim as optim
import random
import torch
import numpy as np

from model import *
from metric import *
from utils import *



SEED = 123

num_neigbor_node = 10
num_paths = 1
depth = 5
num_aspects = 5
h1 =10
sample_num = 1024                                                                                                                                      
neg_sample_num = 128
epochs =1000
user_num = 2005
item_num = 21307

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



rate_matrix = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/rate_matrix.p')
rate_matrix = np.log1p(rate_matrix)
adjacency_matrix = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data//adjacency_matrix.p')
all_adjacency_matrix = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/all_adjacency_matrix.p')
features_user = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/features_user.p')
features_item = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/features_item.p')
features_course = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/features_course.p')
features_teacher = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/features_teacher.p')
features_video = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/features_video.p')
negative = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero//data/negative.p')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
features_user = torch.tensor(features_user, dtype=torch.float32).to(DEVICE)
features_item = torch.tensor(features_item, dtype=torch.float32).to(DEVICE)
features_course = torch.tensor(features_course, dtype=torch.float32).to(DEVICE)
features_teacher = torch.tensor(features_teacher, dtype=torch.float32).to(DEVICE)
features_video = torch.tensor(features_video, dtype=torch.float32).to(DEVICE)

features = torch.cat([features_user,features_item,features_course,features_teacher,features_video])
rating_tensor = torch.tensor(rate_matrix, dtype=torch.float32).to(DEVICE)
adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
negative = torch.tensor(negative,dtype=torch.float32).to(DEVICE)

# Convert to CSR format
adjM = csr_matrix(all_adjacency_matrix, dtype=np.float32)
mooc_adj = adjM+(adjM.T)
training_matrix = sparse.csr_matrix(rate_matrix)
uids, iids = training_matrix.nonzero()
uids_tensor = torch.tensor(np.array(uids), dtype=torch.long)
iids_tensor = torch.tensor(np.array(iids), dtype=torch.long)


dense_matrix = training_matrix.toarray()
tensor_matrix = torch.tensor(dense_matrix)
zero_indices_per_row =[]
for row in tqdm(range(tensor_matrix.size(0))):
    zero_indices = torch.nonzero(tensor_matrix[row] == 0, as_tuple=False).squeeze().tolist()
    neg_indices = negative[row, :-1, 1].tolist()
    unique_indices = [index for index in zero_indices if index not in neg_indices]
    zero_indices_per_row.append(torch.tensor(unique_indices))


node_paths = load_pickle('C:/Users/user/Desktop/Heterogeneous/hetero/data/random_path_10_1_5.p')
node_pairs = node_paths[:,:,[0,-1]].view(-1, 2)
path_tensor = node_paths.clone().detach().to(dtype=torch.long).cuda()

G = create_initial_graph(node_pairs,num_aspects*h1)
G = G.to('cuda:0')
G = dgl.add_self_loop(G)





model =ourmodel(G,features,h1, num_neigbor_node,num_aspects, user_num,item_num,0.2).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=1e-8) 

best_hr_5 = 0.0
best_hr_10 = 0.0
best_hr_20 = 0.0
best_ng_5 = 0.0
best_ng_10 = 0.0
best_ng_20 = 0.0
best_epoch = 0
best_mrr =0
best_auc=0

for epoch in range(epochs):

    train_loss_mean=0
    idx = np.random.randint(low=0, high=len(uids_tensor), size=sample_num)
    batch_u = uids_tensor[idx].reshape(-1, 1).cuda()
    batch_i = iids_tensor[idx].reshape(-1, 1).cuda()
    batch_r= torch.stack([torch.tensor(random.choices(zero_indices_per_row[row.item()], k=neg_sample_num)) for row in batch_u.squeeze()],dim=0).cuda()
    model.train()
    optimizer.zero_grad()

    output, loss, rate_loss = model(path_tensor,batch_u.long(),batch_i.long(),batch_r.long())

    train_loss =  loss  + rate_loss 

    train_loss.backward()
    optimizer.step()
    train_loss_mean +=train_loss.item()
    test_loss  = 0
    test_output = 0
    test_rate_loss = 0
    #if epoch %2 == 0:
    with torch.no_grad():
        model.eval()

        hr_5 = hr(output,negative,user_num,5)
        hr_10 = hr(output,negative,user_num,10)
        hr_20 = hr(output,negative,user_num,20)

        ndcg_5 = ndcg(output,negative,user_num,5)
        ndcg_10 = ndcg(output,negative,user_num,10)
        ndcg_20 = ndcg(output,negative,user_num,20)

        mrr_ = mrr(output,negative,user_num)
        auc_ = auc(output,negative,user_num)

        if ndcg_5 > best_ng_5:
            best_hr_5 = hr_5
            best_hr_10 = hr_10
            best_hr_20 = hr_20
            best_ng_5 = ndcg_5
            best_ng_10 = ndcg_10
            best_ng_20 = ndcg_20
            best_mrr = mrr_
            best_auc = auc_
            best_epoch = epoch
               

    del output
    print('Epoch {:04d}| L1 {:.3f}| L2 {:.3f}|L3 {:.3f}| HR5 {:.3f}| HR10 {:.3f}| HR20 {:.3f}| NDCG5 {:.3f}| NDCG10 {:.3f}| NDCG20 {:.3f}| MRR {:.3f}| AUC {:.3f}'.format(epoch,loss, rate_loss, train_loss, hr_5,hr_10,hr_20,ndcg_5, ndcg_10,ndcg_20,mrr_,auc_))
print('Best Epoch {:05d} | Best HR_5 {:.3f} | Best HR_10 {:.3f} | Best HR_20 {:.3f}  | Best NDCG_5 {:.3f} | Best NDCG_10 {:.3f} | Best NDCG_20 {:.3f}| {:.3f} | {:.3f}| '.format(best_epoch, best_hr_5,best_hr_10,best_hr_20,best_ng_5, best_ng_10,best_ng_20,best_mrr,best_auc))
