import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.parser import parse_args

args = parse_args()


class MIDGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, image_feats, text_feats):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.item_tid_embedding = nn.Embedding(n_items, self.embedding_dim)
        self.item_iid_embedding = nn.Embedding(n_items, self.embedding_dim)
        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.user_preference_image = nn.Embedding(n_users, self.embedding_dim)
        self.user_preference_text = nn.Embedding(n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_tid_embedding.weight)
        nn.init.xavier_uniform_(self.user_preference_image.weight)
        nn.init.xavier_uniform_(self.user_preference_text.weight)
        nn.init.xavier_uniform_(self.item_iid_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        image = self.image_embedding.weight
        text = self.text_embedding.weight
        item_t = self.item_tid_embedding.weight
        item_i = self.item_iid_embedding.weight

        self.fusion_model = IdFusionModel(item_t, item_i, text, image)
        # self.t_gamma = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        # self.i_gamma = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.item_fusion = EmbeddingTwoSemantic(self.embedding_dim)
        self.user_fusion = EmbeddingTwoSemantic(self.embedding_dim)

        # self.tv_fusion = EmbeddingTwoSemantic(self.embedding_dim)
        # self.text2item = nn.Linear(text.shape[1], 128)
        # self.image2item = nn.Linear(image.shape[1], 128)

        self.tau = 0.5

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def forward(self, adj):

        image, item2, text, item1, item_specific_image, item_specific_text, item_fusion_embedding \
            = self.fusion_model()

        # text = self.text2item(self.text_embedding.weight)
        # image = self.image2item(self.image_embedding.weight)
        # item_fusion_embedding = self.tv_fusion(text, image)

        u_ti_embedding = torch.cat((self.user_id_embedding.weight, self.item_tid_embedding.weight), dim=0)
        u_ii_embedding = torch.cat((self.user_id_embedding.weight, self.item_iid_embedding.weight), dim=0)

        ego_embeddings_image = F.normalize(torch.cat((self.user_preference_image.weight, image), dim=0), p=2, dim=1)
        all_embeddings_image = [ego_embeddings_image]
        for i in range(self.n_ui_layers):
            side_embeddings_image = torch.sparse.mm(adj, ego_embeddings_image)
            ego_embeddings_image = side_embeddings_image + args.gamma * u_ii_embedding
            all_embeddings_image += [ego_embeddings_image]
        all_embeddings_image = torch.stack(all_embeddings_image, dim=1)
        all_embeddings_image = all_embeddings_image.mean(dim=1, keepdim=False)
        u_g_embeddings_image, i_g_embeddings_image = torch.split(all_embeddings_image, [self.n_users, self.n_items],
                                                                 dim=0)

        ego_embeddings_text = F.normalize(torch.cat((self.user_preference_text.weight, text), dim=0), p=2, dim=1)
        all_embeddings_text = [ego_embeddings_text]
        for i in range(self.n_ui_layers):
            side_embeddings_text = torch.sparse.mm(adj, ego_embeddings_text)
            ego_embeddings_text = side_embeddings_text + args.gamma * u_ti_embedding
            all_embeddings_text += [ego_embeddings_text]
        all_embeddings_text = torch.stack(all_embeddings_text, dim=1)
        all_embeddings_text = all_embeddings_text.mean(dim=1, keepdim=False)
        u_g_embeddings_text, i_g_embeddings_text = torch.split(all_embeddings_text, [self.n_users, self.n_items],
                                                               dim=0)

        # u_g_embeddings = (u_g_embeddings_text + u_g_embeddings_image) / 2
        u_g_embeddings = self.user_fusion(u_g_embeddings_text, u_g_embeddings_image)
        i_g_embeddings = self.item_fusion(i_g_embeddings_text, i_g_embeddings_image) + \
            F.normalize(item_fusion_embedding, p=2, dim=1)

        return u_g_embeddings, i_g_embeddings, \
            image, item2, text, item1, \
            item_specific_image, item_specific_text, item_fusion_embedding
        # return u_g_embeddings, i_g_embeddings, text, image, item_fusion_embedding


class EmbeddingTwoSemantic(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingTwoSemantic, self).__init__()
        self.query = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_feature1, input_feature2):
        embedding1 = input_feature1
        embedding2 = input_feature2
        att = torch.cat([self.query(embedding1), self.query(embedding2)], dim=-1)
        weight = self.softmax(att)
        h = weight[:, 0].unsqueeze(dim=1) * embedding1 + weight[:, 1].unsqueeze(dim=1) * embedding2

        return h


class IdFusionModel(nn.Module):
    def __init__(self, item1, item2, text, image):
        super(IdFusionModel, self).__init__()
        dim = item1.shape[1]

        self.text2item = nn.Linear(text.shape[1], dim)
        self.image2item = nn.Linear(image.shape[1], dim)

        self.item_embedding1 = item1
        self.item_embedding2 = item2
        self.text_embedding = text
        self.image_embedding = image

        self.text_item = EmbeddingTwoSemantic(dim)
        self.image_item = EmbeddingTwoSemantic(dim)
        self.fusion_item = EmbeddingTwoSemantic(dim)

    def forward(self):
        image = self.image2item(self.image_embedding)
        text = self.text2item(self.text_embedding)
        item1 = self.item_embedding1
        item2 = self.item_embedding2

        item_specific_image = self.image_item(image, item2)
        item_specific_text = self.text_item(text, item1)
        item_fusion_embedding = self.fusion_item(item_specific_image, item_specific_text)

        return image, item2, text, item1, item_specific_image, item_specific_text, item_fusion_embedding

    def get_text(self):
        with torch.no_grad():
            text = self.text2item(self.text_embedding)
        return text

    def get_image(self):
        with torch.no_grad():
            image = self.image2item(self.image_embedding)
        return image


class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        return self.user_embedding.weight, self.item_embedding.weight


class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_ui_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = len(weight_size)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings
