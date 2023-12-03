# IDSF

## Dependencies

- Python 3.6
- torch==1.5.0
- scikit-learn==0.24.2
- torch-scatter==2.0.8

## Dataset Preparation

- Download **5-core reviews data**, **meta data**, and **image features** from [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/links.html). Put data into the directory `data/meta-data/`.

- Install [sentence-transformers](https://www.sbert.net/docs/installation.html) and download [pretrained models](https://www.sbert.net/docs/pretrained_models.html) to extract textual features. Unzip pretrained model into the directory `sentence-transformers/`:

  ```
  ├─ data/: 
      ├── sports/
      	├── meta-data/
      		├── image_features_Sports_and_Outdoors.b
      		├── meta-Sports_and_Outdoors.json.gz
      		├── reviews_Sports_and_Outdoors_5.json.gz
      ├── sentence-transformers/
          	├── stsb-roberta-large
  ```

- Run `python build_data.py` to preprocess data.

- Run `python cold_start.py` to build cold-start data.

## Usage

Start training and inference as:

### Amazon Baby

```shell
python main.py --dataset baby --loss_ratio 0.3 --gamma 0.3
```

### Amazon Sports

```shell
python main.py --dataset sports --loss_ratio 1.0 --gamma 0.3
```

### Amazon Clothing

```shell
python main.py --dataset sports --loss_ratio 1.0 --gamma 1.0
```


## Acknowledgement

Thanks for their works:

> **Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering**
> 
> R. He, J. McAuley
>
> WWW, 2016
>
>[pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/www16a.pdf)

> **Image-based recommendations on styles and substitutes**
>
> J. McAuley, C. Targett, J. Shi, A. van den Hengel
>
> SIGIR, 2015
>
> [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf)
> 

> 
> Latent Structures Mining with Contrastive Modality Fusion for Multimedia
Recommendation. 
> 
> Jinghao Zhang, Yanqiao Zhu, Qiang Liu, Mengqi Zhang, Shu Wu, and Liang Wang. 
> 
> IEEE Transactions on Knowledge and Data Engineering (2023)
