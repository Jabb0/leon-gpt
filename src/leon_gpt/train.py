import time

import composer.optim
import torch.utils.data
from composer import Trainer
from composer.core import DataSpec
from composer.loggers import InMemoryLogger

from leon_gpt.config import TrainerConfig
from leon_gpt.data.telegram_dump import TelegramSingleChatDataset
from leon_gpt.data.tokenizer import SingleCharacterTokenizer
from leon_gpt.models.bigram import BigramLanguageModel


def main(config: TrainerConfig) -> None:

    text_dataset = TelegramSingleChatDataset(
        config.maximum_sequence_length,
        config.dataset_path
    )

    tokenizer = SingleCharacterTokenizer(
        config.max_vocab_size
    )
    tokenizer.fit(text_dataset)

    train_dataloader = torch.utils.data.DataLoader(

    )

    # data_spec = DataSpec()
    #
    model = BigramLanguageModel(
        vocab_size=tokenizer.vocabulary_size(),
        embedding_size=config.embedding_size,
        num_layers=config.num_layers,
        num_heads_per_layer=config.num_heads_per_layer,
        max_sequence_length=config.maximum_sequence_length,
        dropout=config.dropout,
    )

    optimizer = composer.optim.DecoupledSGDW(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = composer.optim.LinearWithWarmupScheduler(
        t_warmup="1ep",  # Warm up over 1 epoch
        alpha_i=1.0,  # Flat LR schedule achieved by having alpha_i == alpha_f
        alpha_f=1.0,
    )

    logger = InMemoryLogger()

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        max_duration=config.max_iterations,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=config.device,
        loggers=logger,
    )

    start_time = time.perf_counter()
    trainer.fit()
    end_time = time.perf_counter()
    print(f"It took {end_time - start_time:0.4f} seconds to train")

    print(trainer.state.train_metrics)
    print(trainer.state.eval_metrics)


if __name__ == '__main__':
    main(TrainerConfig())

