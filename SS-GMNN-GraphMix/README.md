## Training \& Evaluation

**GMNN and GraphMix with self-supervision:**

Go to the corresponding directory and then run the commands. For example:

```
cd ./GMNN-clu/
python run_cora.py
python run_citeseer.py
python run_pubmed.py
```

Our code supports hyper-parameter tuning (grid search) for self-supervision with the following command for example:

```
python run_cora_ss.py
```

## Acknowledgements

The implementations of GMNN and GraphMix are references to [https://github.com/DeepGraphLearning/GMNN](https://github.com/DeepGraphLearning/GMNN) and [https://github.com/vikasverma1077/GraphMix](https://github.com/vikasverma1077/GraphMix).

