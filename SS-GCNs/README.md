## Training \& Evaluation

**GCN and GCN with self-supervision:**

```
python main.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5
python main.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4
python main.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4

python main_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --loss-weight 0.5
python main_clu.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.9
python main_clu.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.9

python main_par.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --partitioning-num 14 --loss-weight 0.7
python main_par.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --partitioning-num 14 --loss-weight 0.8
python main_par.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --partitioning-num 14 --loss-weight 0.2

python main_comp.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --reduced-dimension 48 --loss-weight 0.3
python main_comp.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 24 --loss-weight 0.7
python main_comp.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 28 --loss-weight 0.5
```

Our code supports hyper-parameter tuning (grid search) for self-supervision as stated in the paper. To enable hyper-parameter tuning, run the following command for example:

```
python main_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --grid-search True
```


**GAT \& GIN and GAT \& GIN with self-supervision:**

```
python main_gingat.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --net gin
python main_gingat.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --net gin
python main_gingat.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --net gin

python main_gingat.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --net gat
python main_gingat.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --net gat
python main_gingat.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --net gat

python main_gingat_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --loss-weight 0.7 --net gin
python main_gingat_clu.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.6 --net gin
python main_gingat_clu.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.9 --net gin

python main_gingat_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --loss-weight 0.6 --net gat
python main_gingat_clu.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.3 --net gat
python main_gingat_clu.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.6 --net gat

python main_gingat_par.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --partitioning-num 9 --loss-weight 0.6 --net gin
python main_gingat_par.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --partitioning-num 11 --loss-weight 0.9 --net gin
python main_gingat_par.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --partitioning-num 14 --loss-weight 0.2 --net gin

python main_gingat_par.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --partitioning-num 9 --loss-weight 0.5 --net gat
python main_gingat_par.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --partitioning-num 8 --loss-weight 0.5 --net gat
python main_gingat_par.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --partitioning-num 14 --loss-weight 0.2 --net gat

python main_gingat_comp.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --reduced-dimension 36 --loss-weight 0.3 --net gin
python main_gingat_comp.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 48 --loss-weight 0.5 --net gin
python main_gingat_comp.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 24 --loss-weight 0.3 --net gin

python main_gingat_comp.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --reduced-dimension 24 --loss-weight 0.5 --net gat
python main_gingat_comp.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 24 --loss-weight 0.7 --net gat
python main_gingat_comp.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 24 --loss-weight 0.3 --net gat
```

Hyper-parameter tuning for self-supervision is also supported with the same usage as before.

**cluster_labels** is generated through ```python clu.py``` for node clustering.

## Acknowledgements

The implementations of GCN, GAT and GIN are references to [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn) and [https://github.com/graphdeeplearning/benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns).

