#  [CIKM 2022]-Towards Self-supervised Learning on Graphs with Heterophily
Code of HeteroSSL model proposed in the CIKM 2022 paper [Towards Self-supervised Learning on Graphs with Heterophily](arxiv_url).

## Dependencies

- python 3.9.13
- pytorch 1.12.0
- pytorch-geometric 2.0.4
- scikit-learn 1.1.1
- ogb 1.3.3
- numpy 1.23.1
- munkres 1.1.4
- googledrivedownloader 0.4
- networkx 2.8.5
- matplotlib 3.5.2

## Datasets

Heterogeneous datasets: 'Cornell', Texas', 'Wisconsin', 'Actor', 'Squirrel' and 'Chameleon'.

Homogeneous dataset: 'Cora', 'CiteSeer' and 'PubMed'.

| Dataset          | # Nodes | # Edges | # Classes | # Features | # Homo. ratio |
| ---------------- | ------- | ------- | --------- | ---------- | ------------- |
| Texas            | 183     | 295     | 5         | 1,703      | 0.11          |
| Wisconsin        | 251     | 466     | 5         | 1,703      | 0.21          |
| Actor            | 7,600   | 26,752  | 5         | 931        | 0.22          |
| Squirrel         | 5,201   | 198,493 | 5         | 2,089      | 0.22          |
| Chameleon        | 2,277   | 31,421  | 5         | 2,325      | 0.23          |
| Cornell          | 183     | 280     | 5         | 1,703      | 0.3           |
| CiteSeer         | 3,327   | 4,676   | 7         | 3,703      | 0.74          |
| PubMed           | 19,717  | 44,327  | 3         | 500        | 0.8           |
| Cora             | 2,708   | 5,278   | 6         | 1,433      | 0.81          |


## Usage
To run the codes, use the following commands:
```python
# Cora
python main.py --dataset Cora --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_init_adj --task node_classification --topology_augmentation init

# CiteSeer
python main.py --dataset CiteSeer --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_init_adj --task node_classification --topology_augmentation init

# PubMed
python main.py --dataset PubMed --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_init_adj --task node_classification --topology_augmentation init

# Texas
python main.py --dataset texas --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_learned_adj --task node_classification

# Wisconsin
python main.py --dataset wisconsin --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_learned_adj --task node_classification

# Actor
python main.py --dataset film --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_learned_adj --task node_classification

# Squirrel 
python main.py --dataset squirrel --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_init_adj --task node_classification --topology_augmentation init

# Chameleon
python main.py --dataset squirrel --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.3 --method hn2n_CCA_init_adj --task node_classification --topology_augmentation init

# Cornell
python main.py --dataset cornell --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.4 --method hn2n_CCA_learned_adj --task node_classification

```
For node cluster task, please use "--task node_cluster" and "--output_size 16".

<!-- ## Reference
If our paper and code are useful for your research, please cite the following article:
```
@inproceedings{zhang2021canonical,
  title={From canonical correlation analysis to self-supervised graph neural networks},
  author={Zhang, Hengrui and Wu, Qitian and Yan, Junchi and Wipf, David and Philip, S Yu},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
``` -->