import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from bert import BertForPreTraining
from transformers.pytorch.bert.data import (
    eval_input_dataloader,
    train_input_dataloader,
)
from transformers.pytorch.bert.utils import set_defaults
from pytorch.run_utils import run

def main():

    run(
        BertForPreTrainingModel,
        train_input_dataloader,
        eval_input_dataloader,
        set_defaults,
    )


if __name__ == '__main__':
    main()
