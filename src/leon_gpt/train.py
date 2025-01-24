import torch.utils.data
from composer import Trainer
from composer.core import DataSpec

from leon_gpt.config import TrainerConfig
from leon_gpt.data.telegram_dump import TelegramSingleChatDataset
from leon_gpt.data.tokenizer import SingleCharacterTokenizer


def main(config: TrainerConfig) -> None:

    text_dataset = TelegramSingleChatDataset(
        config.maximum_sequence_length,
        config.dataset_path
    )

    tokenizer = SingleCharacterTokenizer(
        config.max_vocab_size
    )
    tokenizer.fit(text_dataset)

    train_dataloader = torch.utils.data.DataLoader()

    # data_spec = DataSpec()
    #
    # model =
    #
    # trainer = Trainer(
    #     train_dataloader=data_spec,
    #
    # )



if __name__ == '__main__':
    main(TrainerConfig())

