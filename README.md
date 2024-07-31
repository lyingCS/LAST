# LAST

LAST is an extended version of [LibRerank](https://github.com/LibRerank-Community/LibRerank) 

## Quick Started

### Install repo from source

```
git clone https://github.com/lyingCS/LAST.git
cd LAST
make init 
```

### Decompress evaluator checkpoint

For facilitate the training of the generator, we provide a  version of the checkpoints of CMR_evaluator that have been pretrained. We first need to decompress it.

```
tar -xzvf ./model/save_model_ad/10/*.tar.gz -C ./model/save_model_ad/10/
```

### Train LAST

```
python run_reranker.py
```

Model parameters can be set by using a config file, and specify its file path at `--setting_path`, e.g., `python run_ranker.py --setting_path config`. The config files for the different models can be found in `example/config`. Moreover, model parameters can also be directly set from the command line.

### Eval LAST model

```
python eval_last_model.py --reload_path='path/to/model/you/trained'
```

**For more information please refer to [LibRerank_README.md](./LibRerank_README.md)**

## Citation

Please cite our paper if you use this repository.

```
@article{wang2024not,
  title={Do Not Wait: Learning Re-Ranking Model Without User Feedback At Serving Time in E-Commerce},
  author={Wang, Yuan and Li, Zhiyu and Zhang, Changshuo and Chen, Sirui and Zhang, Xiao and Xu, Jun and Lin, Quan},
  journal={arXiv preprint arXiv:2406.14004},
  year={2024}
}
```
