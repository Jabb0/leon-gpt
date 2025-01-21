# Leon-GPT

This repository mimics Leon using a really simple GPT implementation from scratch.

Adapted from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).

## Run
```bash
# Get all system and python dependencies set up.
nix develop
# Run pycharm
pycharm-community . &> /dev/null &
# Download the dataset
cd data && wget https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt
python main.py
```



## Milestones
1. Train the model as done by Andrej on Shakespear.
2. Train the model on signal chat with Leon.
3. Use open source models for this.

### Tech Considerations
1. Use DVC for model and data management + reproducibility.
2. Use composer for training.
3. Use hydra for hyperparameter configuration.
4. Use WnB for logging.

