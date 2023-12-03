import math
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import MIDGCN, LightGCN
# from Decouple import MIDGCN
from utility.batch_test import *
from utility.logging import Logger
from utility.parser import parse_args

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.task_name = "%s_%s_%s_%s" % (
        datetime.now().strftime('%m-%d %H-%M'), args.dataset, args.loss_ratio, args.gamma)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("task_name: %s" % self.task_name)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().cuda()

        image_feats = np.load('./data/{}/image_feat.npy'.format(args.dataset))
        text_feats = np.load('./data/{}/text_feat.npy'.format(args.dataset))

        self.model = MIDGCN(self.n_users, self.n_items, self.emb_dim, self.weight_size, image_feats, text_feats)
        # self.model = LightGCN(self.n_users, self.n_items, self.emb_dim, self.weight_size, None, None, None)

        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model(self.norm_adj)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1

                ua_embeddings, ia_embeddings, \
                    image, item2, text, item1, \
                    item_specific_image, item_specific_text, item_fusion_embedding \
                    = self.model(self.norm_adj)
                # ua_embeddings, ia_embeddings, text, image, item_fusion_embedding = self.model(self.norm_adj)
                # ua_embeddings, ia_embeddings = self.model(self.norm_adj)

                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_contrastive_loss = 0

                # cont2. ***
                batch_contrastive_loss += self.model.batched_contrastive_loss(image, item_specific_image)
                batch_contrastive_loss += self.model.batched_contrastive_loss(item2, item_specific_image)
                batch_contrastive_loss += self.model.batched_contrastive_loss(text, item_specific_text)
                batch_contrastive_loss += self.model.batched_contrastive_loss(item1, item_specific_text)
                batch_contrastive_loss += self.model.batched_contrastive_loss(item_specific_image,
                                                                              item_fusion_embedding)
                batch_contrastive_loss += self.model.batched_contrastive_loss(item_specific_text, item_fusion_embedding)

                # batch_contrastive_loss += self.model.batched_contrastive_loss(text, item_fusion_embedding)
                # batch_contrastive_loss += self.model.batched_contrastive_loss(image, item_fusion_embedding)

                batch_contrastive_loss *= args.loss_ratio
                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_contrastive_loss

                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
                contrastive_loss += float(batch_contrastive_loss)

            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f" % (eval(args.Ks)[1], test_ret['recall'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break

        self.logger.logging(str(test_ret))
        torch.save(self.model.item_tid_embedding.weight, 'ids/IDSF/item_tid_clothing.pt')
        torch.save(self.model.item_iid_embedding.weight, 'ids/IDSF/user_iid_clothing.pt')
        torch.save(self.model.fusion_model.get_text(), 'ids/IDSF/item_text_clothing.pt')
        torch.save(self.model.fusion_model.get_image(), 'ids/IDSF/item_image_clothing.pt')

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj

    trainer = Trainer(data_config=config)
    trainer.train()
