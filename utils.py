from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import random
import pickle
import torch
import dgl 
SEED = 123

# PyTorch
np.random.seed(SEED)
random.seed(SEED)

def get_non_zero_neighbors(matrix: csr_matrix, node,same_node, exclude_range,sample_size=None,stop=False):
    start_ptr, end_ptr = matrix.indptr[node], matrix.indptr[node + 1]
    neighbors = matrix.indices[start_ptr:end_ptr]
    st,end = exclude_range
    if stop :
        neighbors  = [value for value in neighbors if value in range(st,end)]
    else:
        neighbors  = [value for value in neighbors if value  not in range(st,end)]   
    if sample_size is not None and len(neighbors) > sample_size:
        #neighbors  = [value for value in neighbors if value not in exclude_range]
        neighbors = np.random.choice(neighbors, sample_size, replace=False)
  
    return neighbors

def random_walk(matrix, start_node, walk_length,exclude_range, sample_size=10):
    walk = [start_node]
    current_node = start_node
    stop =False
    for i in range(walk_length - 1):
        if i == walk_length-1:
            stop = True
        neighbors = get_non_zero_neighbors(matrix, current_node, walk,exclude_range, sample_size,stop)

        if len(neighbors) == 0:
            break  # 이동할 이웃 노드가 없으면 종료

        next_node = random.choice(neighbors)
        walk.append(next_node)
        current_node = next_node

    return walk




def bidirectional_search1(graph, start, end, depth, iteration,exclude_range):
    forward_steps = depth // 2
    backward_steps = depth // 2
    if depth % 2 == 0:
        forward_steps += 1
    else : 
        forward_steps += 1
        backward_steps += 1

    forward_path = np.empty(( 0,forward_steps), dtype=object)
    backward_path = np.empty((0, backward_steps), dtype=object)
    concat_list = []
    n = 0
    while len(concat_list) == 0 and n < 10:
        try:
            for i in range(iteration):
                
                forward_path = np.append(forward_path, [random_walk(graph, start, forward_steps, exclude_range)], axis=0)
                backward_path = np.append(backward_path, [random_walk(graph, end, backward_steps, exclude_range)], axis=0)

            forward_path_end = forward_path[:,-1]
            backward_path_end = backward_path[:,-1]

            matching_pairs = np.array([[idx_a, idx_b] for idx_a, node_a in enumerate(forward_path_end) for idx_b, node_b in enumerate(backward_path_end) if node_a == node_b])


            if matching_pairs.shape[0] > 0:

                for_list_b = forward_path[matching_pairs[:,0]]
                back_list_b = backward_path[matching_pairs[:,1]][:,:-1][:, ::-1]
                
                concat_list = list(np.concatenate([for_list_b,back_list_b],axis=1)[0])
        except ValueError:
            pass        
        n += 1
    return concat_list,n


    

def guided_path_walk(mooc_adj,num_neigbor_node,num_paths,depth,data,MAX_TRIES = 50):
    node_paths = []
    skip_node = []
    user = 16239
    item = 14284
    for i in range(mooc_adj.shape[0]):
        if len(mooc_adj[i].nonzero()[0]) == 0 :
            skip_node.append(i)
    for start_node in tqdm(range(0,item+user)):
        node_path = []
        if start_node > user:
            end_node_range = (user, item-1)
        else:
            end_node_range = (0, user-1)

        tries = 0
        selected_end_nodes = set()  # 이미 선택된 end_node를 기록하기 위한 집합ta
        if start_node not in skip_node:
            while len(node_path) < num_neigbor_node and tries < MAX_TRIES:

                    end_node = random.randint(*end_node_range)
                    if start_node != end_node and end_node not in selected_end_nodes:
                
                            nodes= random_walk(mooc_adj, start_node, depth,end_node_range)

                           # print(nodes,i)
                            if len(nodes) == depth :
                                node_path.extend(nodes)
   
                                selected_end_nodes.add(end_node)  # 선택된 end_node를 기록
            
                    tries += 1
  

        else:
            node_path = [[start_node]*depth]*num_neigbor_node
            print(start_node,'SKIP')
        node_paths.append(node_path)


    save_pickle(node_paths,f'data/{data}/random_path_{num_neigbor_node}_{num_paths}_{depth}.p')
    return node_paths





def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def guided_path_walk(g,num_neigbor_node,num_paths,depth,data,MAX_TRIES = 50):
    node_paths = []
    skip_node = []
    user = 16239
    item = 14284
    for start_node in tqdm(range(0,user+item)):
        node_walk = []
        while len(node_walk) < 10 :
            if start_node > user:
                end_node_range = (user, item-1)
            else:
                end_node_range = (0, user-1)
            st,end = end_node_range
            traces, types = dgl.sampling.random_walk(g, start_node, length=4)
            traces = traces.tolist()[0]
            if traces[-1] in range(st,end):
                include = True
                for item in traces[1:-1]:
                    if item in range(st,end):
                        include = False
                if  include == True:        
                    node_walk.append(traces)   
        node_paths.append(node_walk)

    save_pickle(node_paths,f'data/{data}/random_path_{num_neigbor_node}_{num_paths}_{depth}.p')
    return node_paths


def path_pair(user_path):
    user_start = user_path[:, :, 0].reshape(-1, 1)
    user_end = user_path[:, :, -1].reshape(-1, 1)
    user_path_pair = torch.cat((user_start, user_end), dim=1)
    return user_path_pair

def update_edge_weights(G, model_node_pairs, new_weights):
    src_nodes = model_node_pairs[:, 0]
    dst_nodes = model_node_pairs[:, 1]
    edges = G.edge_ids(src_nodes, dst_nodes)
    G.edata['weight'][edges] = torch.tensor(new_weights)


def create_initial_graph(node_pairs,aspect_num):
    src_nodes, dst_nodes = zip(*node_pairs)
    G = dgl.graph((src_nodes, dst_nodes))
    G.edata['weight'] = torch.ones(len(node_pairs),aspect_num)
    return G