## Dependencies

TBD

## Training \& Evaluation

**GCN and GCN with self-supervision:**

```shell
python main.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5
python main.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4
python main.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4

python main_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --loss-weight 0.5
python main_clu.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --loss-weight
python main_clu.py --dataset pubmed --embedding-dim 500 16 3 --lr 0.01 --weight-decay 5e-4 --loss-weight
```

TBD

