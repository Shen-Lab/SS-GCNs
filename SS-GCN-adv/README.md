## Training \& Evaluation

**GCN under adversarial attacks:**

```
python main_attack.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --nattack 2
python main_attack.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --nattack 2
```

**GCN with adversarial training under adversarial attacks:**

```
python main_defense.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --nattack 2
python main_defense.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --nattack 2
```

**GCN with self-supervised adversarial training under adversarial attacks:**

```
python main_defense_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --task-ratio 0.5 --nattack 2
python main_defense_clu.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --task-ratio 0.9 --nattack 2

python main_defense_par.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --partition-num 14 --task-ratio 0.7 --nattack 2
python main_defense_par.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4 --partition-num 14 --task-ratio 0.8 --nattack 2

python main_defense_comp.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --partition-num 48 --task-ratio 0.3 --nattack 2
python main_defense_comp.py --dataset citeseer --embedding-dim 3703 16 6 --lr 0.01 --weight-decay 5e-4--partition-num 24 --task-ratio 0.7 --nattack 2
```

The attack strength can be altered via setting ```--nattack```. For example:

```
python main_defense_clu.py --dataset cora --embedding-dim 1433 16 7 --lr 0.008 --weight-decay 8e-5 --task-ratio 0.5 --nattack 3
```

## Acknowledgements

The attack algorithm, Nettack is reference to [https://github.com/danielzuegner/nettack](https://github.com/danielzuegner/nettack).

