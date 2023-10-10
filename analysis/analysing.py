from utils import sample, get_feature, get_heatmap, ground_truth, get_scatter
import torch
import random

"""
modals = ["gt", "id", "iid", "tid", "image", "text", "text_ft", "image_ft"]
data_types = ["train", "val", "test"]
datasets = ["clothing", "baby", "sports"]
painters = ["scatter", "heatmap"]
"""
DATASET = "clothing"
MODAL = "image"
DATA_TYPE = "train"
BATCH_SIZE = 10
K = 10
PAINTER = "heatmap"
METHOD = "umap"
DIM = 2
RANDOM_SEED = 538
random.seed(RANDOM_SEED)


def main():
    filename = "{0}_{1}".format(DATASET, MODAL)

    user_batch, item_batch, labels, init_index = sample(DATASET, DATA_TYPE, BATCH_SIZE)
    print(user_batch, len(item_batch))

    user_batch, item_batch = torch.LongTensor(user_batch), torch.LongTensor(item_batch)

    if MODAL == 'gt':
        ground_truth(item_batch, filename)
    else:
        item_feature, user_feature = get_feature(DATASET, MODAL)
        if PAINTER == "heatmap":
            get_heatmap(item_batch, item_feature, K, filename)
        elif PAINTER == "scatter":
            assert METHOD is not None
            assert DIM is not None
            get_scatter(item_batch, item_feature, labels, filename, method=METHOD, dim=DIM)

    print("Done.")


if __name__ == "__main__":
    main()
