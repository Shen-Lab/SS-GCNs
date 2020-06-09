## Dependencies

TBD

## Training \& Evaluation

**GCN and GCN with self-supervision:**

```shell
python main.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5
python main.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4
python main.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4

python main_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --loss-weight 0.5
python main_clu.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.9
python main_clu.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --loss-weight 0.9

python main_par.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --partition-num 14 --loss-weight 0.7
python main_par.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --partition-num 14 --loss-weight 0.8
python main_par.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --partition-num 14 --loss-weight 0.2

python main_comp.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --reduced-dimension 48 --loss-weight 0.3
python main_comp.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 24 --loss-weight 0.7
python main_comp.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --reduced-dimension 28 --loss-weight 0.5


```

TBD

