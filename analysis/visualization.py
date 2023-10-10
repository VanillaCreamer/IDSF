import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn import manifold
import seaborn as sns
import torch


class Reduction(object):
    def __init__(self):
        self.reducer_2d = None
        self.reducer_3d = None
        self.file_name = None
        self.postfix = None
        # plt.axes().get_xaxis().set_visible(False)
        # plt.axes().get_yaxis().set_visible(False)
        # plt.xticks([])
        # plt.yticks([])

    def painting_2d(self, data, label: list[int], filename):
        embedding = self.reducer_2d.fit_transform(data.cpu())
        x = embedding[:, 0]
        y = embedding[:, 1]
        color = [sns.husl_palette(10)[i] for i in label]
        plt.xticks([])
        plt.yticks([])
        plt.scatter(x, y, s=100, c=color)
        plt.savefig('scatter/2d/{0}_{1}_2d.svg'.format(filename, self.postfix), dpi=512)

    def painting_3d(self, data, label: list[int], filename):
        embedding = self.reducer_3d.fit_transform(data.cpu())
        x = embedding[:, 0]
        y = embedding[:, 1]
        z = embedding[:, 2]
        color = [sns.husl_palette(10)[i] for i in label]

        ax = plt.axes(projection='3d')
        # hide x,y,z ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        ax.scatter(x, y, z, s=100, c=color)
        plt.savefig('scatter/3d/{0}_{1}_3d.svg'.format(filename, self.postfix), dpi=512)


class UMAPReduction(Reduction):
    def __init__(self):
        super().__init__()
        self.reducer_2d = umap.UMAP(n_components=2)
        self.reducer_3d = umap.UMAP(n_components=3)
        self.postfix = 'UMAP'


class TSNEReduction(Reduction):
    def __init__(self):
        super().__init__()
        self.reducer_2d = manifold.TSNE(n_components=2, init='pca')
        self.reducer_3d = manifold.TSNE(n_components=3, init='pca')
        self.postfix = 'TSNE'


def sns_mapping(matrix, batch_id, filename):
    colors = ["white", "lemonchiffon", "c", "darkblue"]
    nodes = [0.0, 0.1, 0.4, 1.0]
    my_cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    fig = sns.heatmap(matrix, square=True,  cmap="YlGnBu")  # linecolor="black", linewidths=1)  # , cbar=False)  # yticklabels=batch_id.tolist() xticklabels=[], yticklabels=[],
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)
    heatmap = fig.get_figure()
    heatmap.savefig("heatmap/{0}.svg".format(filename), dpi=1080)


class Heatmapper(object):
    def __init__(self, k=None):
        self.k = k

    def sim_mapping(self, matrix1, matrix2, batch_id, filename):
        def cal_similarity(m1, m2):
            m1 = m1 / torch.norm(m1, dim=-1, keepdim=True)
            m2 = m2 / torch.norm(m2, dim=-1, keepdim=True)
            sim = torch.mm(m1, m2.T)
            # sim = sim / torch.norm(sim, dim=1, keepdim=True)
            return sim

        def top_k(sim):
            topk_values, topk_indices = torch.topk(sim, k=self.k, dim=1)
            sparse_tensor = torch.zeros_like(sim)
            for i in range(sim.shape[0]):
                sparse_tensor[i][topk_indices[i]] = topk_values[i]

            # put diagonal as 0
            # diagonal = np.diag_indices(sparse_tensor.shape[0])
            # sparse_tensor[diagonal] = 0

            return sparse_tensor

        if self.k is None:
            sim_mat = cal_similarity(matrix1, matrix2).cpu().numpy()
        else:
            sim_mat = top_k(cal_similarity(matrix1, matrix2)).cpu().numpy()
        # sim_mat = self.sort_sim(self.cal_similarity()).cpu().numpy()
        sns_mapping(sim_mat, batch_id, filename)
