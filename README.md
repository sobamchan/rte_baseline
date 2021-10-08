# RTE baseline

Simple, clean implementation to finetune BERT on RTE (paraphrase) dataset.


# Data format

Json Lines format, which has three keys, s1 (string), s2 (string) and label (boolean).

```
{"s1": "First sentence.", "s2": "Second sentence.", "label": 0 or 1}
```


# Run

This repo contains MRPC (binary paraphrase task) as an example. You can add your dataset as long as it follows the format above.
It uses poetry as a package manager, please refer to [the official documentation](https://python-poetry.org/) for its installation.

## set up
```bash
> poetry install  # Installs all the requirements.
> poetry shell  # Login to the environment.
```

```bash
# checkout exec command options
> python src/main.py -h

# start training
> python src/main.py --lr 1e-5 --batch-size 8 --max-epochs 5 --seed 42 --dpath ./data/ --default-root-dir ./ --gpus 0
```

It's built on top of huggingface and pytorch-lightning, so the command above save training checkpoint which later you can use for inference. For now, it writes log in format of tensorboard so during or after the training you can check the progress by running a command below.

```bash
> tensorboard --logdir lightning_logs
```


# ToDos

- [ ] Add evaluation example
- [ ] Add inference example
- [ ] Proper early stopping
- [ ] Cool logging
