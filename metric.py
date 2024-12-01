import torch 

def ndcg(rate, negative, length, k=5):
    negative = negative.long()
    # rate에서 negative 인덱스에 해당하는 값을 가져온다.
    test = rate[negative[:, :, 0], negative[:, :, 1]]

    # 상위 k개의 인덱스를 구한다.
    topk_values, topk_indices = torch.topk(test, k=k, dim=1, largest=True, sorted=True)

    # 99와 일치하는 인덱스를 찾는다.
    n = (topk_indices == 99).nonzero(as_tuple=False)[:, 1]

    # NDCG를 계산한다.
    ndcg_score = torch.sum(torch.log(torch.tensor(2.0)) / torch.log(n.to(torch.float32) + 2)) / length

    return ndcg_score


def auc(rate, negative, length):
    negative = negative.long()
    test = rate[negative[:, :, 0], negative[:, :, 1]]
    topk = torch.topk(test, 100).indices
    where = (topk == 99).nonzero(as_tuple=False)
    auc = where[:, 1]
    ran_auc = torch.randint(0, 100, (length, 1), dtype=torch.int64)
    auc = torch.mean((auc - ran_auc.cuda() < 0).float())
    return auc

def hr(rate, negative, length, k=5):
    negative = negative.long()
    test = rate[negative[:, :, 0], negative[:, :, 1]]
    topk = torch.topk(test, k).indices
    isIn = (topk == 99).float()
    row = torch.sum(isIn, dim=1)
    all = torch.sum(row)
    return all / length

def mrr(rate, negative, length):
    negative = negative.long()
    test = rate[negative[:, :, 0], negative[:, :, 1]]
    topk = torch.topk(test, 100).indices
    mrr_ = torch.sum(1.0 / ((topk == 99).nonzero(as_tuple=False)[:, 1].float() + 1.0))
    mrr_value = mrr_ / length
    return mrr_value
