# Leon-GPT

This repository mimics Leon using a really simple GPT implementation from scratch.

Adapted from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).

This repo is a playground for training ML models with data and model management, experiment training and logging.

There exists the [llm-foundry](https://github.com/mosaicml/llm-foundry/tree/main) project for production system reference of LLM systems.

### Tech-Stack
- [Composer](https://github.com/mosaicml/composer) for training orchestration.
- PyTorch
- 

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

## Data preparation
Given a telegram dump file this extract all messages:
```bash
jq `keys` file.json  # To check what is in there.
jq '[.messages[] | select(.text != "") | {from , text}]' file.json > new_file.json

```


## Milestones
- [x] Train the model as done by Andrej on Shakespear.
- [] Train the model on chats with Leon.
- [] Use open source models for this.

### Tech Considerations
1. Use DVC for model and data management + reproducibility.
2. Use composer for training.
3. Use hydra for hyperparameter configuration.
4. Use WnB for logging.
5. Adding gitlab CI pipeline.

### Testing
```bash
pytest
```
