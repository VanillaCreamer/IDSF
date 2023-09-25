import json
import torch
import itertools
import numpy as np
import random
from visualization import Heatmapper, sns_mapping, UMAPReduction, TSNEReduction
from adj import Data


def sample(dataset, data_type, batch_size):
    """
    :param dataset: dataset name
    :param batch_size: the number of users to sampling
    :param data_type: string with [train, val, test] e.g. "train_test"
    :return: sampled users' id, list of all items, labels of each item, index of each item
    """
    user_item_train = json.load(open('../data/{0}/5-core/train.json'.format(dataset)))
    user_item_val = json.load(open('../data/{0}/5-core/val.json'.format(dataset)))
    user_item_test = json.load(open('../data/{0}/5-core/test.json'.format(dataset)))
    users = [int(k) for k in user_item_train.keys()]
    user_sample = random.sample(users, batch_size)
    item_sample = []
    types = data_type.split("_")
    labels = []
    index = []
    for u in user_sample:
        index.append(len(labels))
        item_sample.append(user_item_train[str(u)])
        for i in range(len(user_item_train[str(u)])):
            labels.append(user_sample.index(u))
        if "val" in types:
            item_sample.append(user_item_val[str(u)])
            for i in range(len(user_item_val[str(u)])):
                labels.append(user_sample.index(u))
        if "test" in types:
            item_sample.append(user_item_test[str(u)])
            for i in range(len(user_item_test[str(u)])):
                labels.append(user_sample.index(u))

    return user_sample, list(itertools.chain(*item_sample)), labels, index  # 将内嵌列表平铺


def get_feature(dataset, modal):
    """
    :param dataset: dataset name ["clothing", "baby", "sports"]
    :param modal: modality from [id, text, image, text_ft]
    :return:
    """
    item_feature, user_feature = None, None
    if modal == "id":
        item_feature = torch.load("../ids/LightGCN/item_id_{0}.pt".format(dataset)).data
        user_feature = torch.load("../ids/LightGCN/item_id_{0}.pt".format(dataset)).data
    elif modal == "image":
        item_feature = torch.tensor(np.load('../data/{0}/image_feat.npy'.format(dataset))).data
    elif modal == "text":
        item_feature = torch.tensor(np.load('../data/{0}/text_feat.npy'.format(dataset))).data
    elif modal == "text_ft":
        item_feature = torch.load("../ids/IDSF/item_text_{0}.pt".format(dataset)).data
    elif modal == "image_ft":
        item_feature = torch.load("../ids/IDSF/item_image_{0}.pt".format(dataset)).data
    elif modal == "iid":
        item_feature = torch.load("../ids/IDSF/item_iid_{0}.pt".format(dataset)).data
    elif modal == "tid":
        item_feature = torch.load("../ids/IDSF/item_tid_{0}.pt".format(dataset)).data
    return item_feature, user_feature


def get_heatmap(batch_id, feature, topk, filename):

    batch_feature = feature[batch_id]
    model = Heatmapper(topk)
    model.sim_mapping(batch_feature, batch_feature, batch_id, filename)


def ground_truth(item_batch, filename):
    dataset = Data()
    num_user = dataset.n_users
    num_item = dataset.n_items

    def get_adj_mat():
        try:
            adj_mat, norm_adj_mat, mean_adj_mat = dataset.get_adj_mat()
            print('already load adj matrix')
        except Exception:
            raise NotImplementedError
        return adj_mat, norm_adj_mat, mean_adj_mat

    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    adj, norm_adj, mean_adj = get_adj_mat()
    adj_sparse = sparse_mx_to_torch_sparse_tensor(adj).cuda()
    # adj_sparse = adj.cuda()
    s2 = adj_sparse + adj_sparse @ adj_sparse
    item2item = s2.to_dense()[num_user:, num_user:]

    # index sampled items
    x, y = torch.meshgrid(item_batch, item_batch)
    indices = torch.stack((x, y), dim=-1)
    index = torch.reshape(indices, (-1, 2))
    row = index[:, 0]
    col = index[:, 1]

    sim_mx = item2item[row, col].reshape(item_batch.shape[0], item_batch.shape[0])
    sim_mx = torch.where(sim_mx != 0, torch.tensor(1), sim_mx)
    sim_mx = sim_mx.cpu().numpy()

    sns_mapping(sim_mx, item_batch, filename)


def get_scatter(batch_id, feature, labels, filename, method=None, dim=None):
    painter = None
    batch_feature = feature[batch_id]
    if method == "tsne":
        painter = TSNEReduction()
    elif method == "umap":
        painter = UMAPReduction()
    else:
        raise NotImplementedError

    if dim == 2:
        painter.painting_2d(batch_feature, labels, filename)
    elif dim == 3:
        painter.painting_3d(batch_feature, labels, filename)
    else:
        raise NotImplementedError

