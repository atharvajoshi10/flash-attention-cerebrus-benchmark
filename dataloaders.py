import sys
import csv

import numpy as np
import torch
from utils import(
    build_vocab,
    create_masked_lm_predictions,
    get_meta_data,
    parse_text,
    shard_and_shuffle_data,
)

import random

import numpy as np
from torch.utils.data._utils.collate import default_collate
from torch.distributed import get_rank, get_world_size, is_initialized

def get_data_for_task(
    task_id,
    meta_data_values_cum_sum,
    num_examples_per_task,
    meta_data_values,
    meta_data_filenames,
):
    """
    Function to get distribute files with given number of examples such that each
    distributed task has access to exactly the same number of examples

    Args:
        task_id (int): Integer id for a task.
        meta_data_values_cum_sum (int): Cumulative sum of the file sizes in 
            lines from meta data file.
        num_examples_per_task (int): Number of the examples specified per  
            slurm task. Equal to `batch_size` * `num_batch_per_task`.
        meta_data_values (list[int]): List of the files sizes in lines in the 
            meta data file.
        meta_data_filenames (list[str]): List with file names in the meta data 
            file.

    Returns:
        list of tuples of length 3. The tuple contains at
        - index 0: filepath.
        - index 1: number of examples to be considered for this task_id.
        - index 2: start index in the file from where these
                    examples should be considered
        The list represents the files that should be considered for this task_id.
    """
    files_in_task = []

    # file where the split starts
    file_start_idx = np.min(
        np.where(meta_data_values_cum_sum > task_id * num_examples_per_task)[0]
    )
    # Index in file from where the examples should be considered for this task
    start_idx = (
        task_id * num_examples_per_task
        - meta_data_values_cum_sum[file_start_idx - 1]
        # -1 since len(`meta_data_values_cum_sum`) = len(`meta_data_values`) + 1
    )

    # Number of examples to pick from this file.
    # We do a `min` to handle a case where the file has
    # examples > num_examples_per_task
    num_examples = min(
        meta_data_values[file_start_idx - 1] - start_idx, num_examples_per_task,
    )
    files_in_task.append(
        (
            meta_data_filenames[file_start_idx - 1],
            num_examples,
            start_idx,
        )  # (file_path, num_examples, start_index)
    )

    if num_examples != num_examples_per_task:
        # If the file has fewer number of examples than
        # `num_examples_per_task`, continue through files
        # till we reach our required number of examples.

        indices = np.where(
            meta_data_values_cum_sum > (task_id + 1) * num_examples_per_task
        )[0]
        if indices.size != 0:
            file_end_idx = np.min(indices)
        else:
            file_end_idx = len(meta_data_values_cum_sum)

        for i in range(file_start_idx + 1, file_end_idx):
            files_in_task.append(
                (
                    meta_data_filenames[i - 1],
                    meta_data_values[i - 1],
                    0,
                )  # (file_path, num_examples, start_index)
            )

        # If the number of examples needed to fulfill
        # `num_examples_per_task`, falls in between a file
        num_end_examples = (
            task_id + 1
        ) * num_examples_per_task - meta_data_values_cum_sum[file_end_idx - 1]
        if num_end_examples > 0:
            files_in_task.append(
                (
                    meta_data_filenames[file_end_idx - 1],
                    num_end_examples,
                    0,
                )  # (file_path, num_examples, start_index)
            )

    assert (
        sum([num_examples for _, num_examples, _ in files_in_task])
        == num_examples_per_task
    ), f"Incorrect number of examples in the split with task_id {task_id}"

    return files_in_task


def bucketed_batch(
    data_iterator,
    batch_size,
    buckets=None,
    element_length_fn=None,
    collate_fn=None,
    drop_last=False,
    seed=None,
):
    """
    Batch the data from an iterator such that sampels of similar length end
    up in the same batch. If `buckets` is not supplied, then this just batches
    the dataset normally.

    :param data_iterator: An iterater that yields data one sample at a time.
    :param int batch_size: The number of samples in a batch.
    :param list buckets: A list of bucket boundaries. If set to None, then no
        bucketing will happen, and data will be batched normally. If set to
        a list, then data will be grouped into `len(buckets) + 1` buckets. A
        sample `s` will go into bucket `i` if
        `buckets[i-1] <= element_length_fn(s) < buckets[i]` where 0 and inf are
        the implied lowest and highest boundaries respectively. `buckets` must
        be sorted and all elements must be non-zero.
    :param callable element_length_fn: A function that takes a single sample and
        returns an int representing the length of that sample.
    :param callable collate_fn: The function to use to collate samples into a
        batch. Defaults to PyTorch's default collate function.
    :param bool drop_last: Whether or not to drop incomplete batches at the end
        of the dataset. If using bucketing, buckets that are not completely full
        will also be dropped, even if combined there are more than `batch_size`
        samples remaining spread across multiple buckets.
    :param int seed: If using `drop_last = False`, we don't want to feed out
        leftover samples with order correlated to their lengths. The solution
        is to shuffle the leftover samples before batching and yielding them.
        This seed gives the option to make this shuffle deterministic. It is
        only used when `buckets` is not `None` and `drop_last = True`.

    :yields: Batches of samples of type returned by `collate_fn`, or batches of
        PyTorch tensors if using the default collate function.
    """
    if batch_size < 1:
        raise ValueError(f"Batch size must be at least 1. Got {batch_size}.")

    if collate_fn is None:
        collate_fn = default_collate
    rng = random.Random(seed)

    if buckets is None:
        # batch the data normally
        batch = []
        for element in data_iterator:
            batch.append(element)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
        if not drop_last and len(batch) > 0:
            yield collate_fn(batch)

    elif isinstance(buckets, list):
        if sorted(buckets) != buckets:
            raise ValueError(
                f"Bucket boundaries must be sorted. Got {buckets}."
            )
        if buckets[0] <= 0:
            raise ValueError(
                f"Bucket boundaries must be greater than zero. Got {buckets}."
            )
        if not isinstance(buckets[0], int):
            t = type(buckets[0])
            raise ValueError(
                f"Elements of `buckets` must be integers. Got {t}."
            )
        if element_length_fn is None:
            raise ValueError(
                "You must supply a length function when using bucketing."
            )

        bucket_contents = [[] for i in range(len(buckets) + 1)]
        for element in data_iterator:
            length = element_length_fn(element)
            bucket_index = np.searchsorted(buckets, length, side="right")
            bucket_contents[bucket_index].append(element)
            if len(bucket_contents[bucket_index]) == batch_size:
                yield collate_fn(bucket_contents[bucket_index])
                bucket_contents[bucket_index] = []

        if not drop_last:
            remaining_data = []
            for bucket in bucket_contents:
                remaining_data.extend(bucket)
            rng.shuffle(remaining_data)
            batch = []
            for element in remaining_data:
                batch.append(element)
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch)

    else:
        raise ValueError(
            "buckets must be None or a list of boundaries. " f"Got {buckets}."
        )


class BertCSVDynamicMaskDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads csv files containing the input text tokens, adds MLM features
    on the fly.
    :param <dict> params: dict containing input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (string): path to the data files to use.
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "repeat" (bool): Flag to enable data repeat.
    - "mask_whole_word" (bool): Flag to whether mask the entire word.
    - "do_lower" (bool): Flag to lower case the texts.
    - "dynamic_mlm_scale" (bool): Flag to dynamically scale the loss.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    - "oov_token" (string): Out of vocabulary token.
    - "mask_token" (string): Mask token.
    - "document_separator_token" (string): Seperator token.
    - "exclude_from_masking" list(string): tokens that should be excluded from being masked.
    - "max_sequence_length" (int): Maximum length of the sequence to generate.
    - "max_predictions_per_seq" (int): Maximum number of masked tokens per sequence.
    - "masked_lm_prob" (float): Ratio of the masked tokens over the sequence length.
    - "gather_mlm_labels" (bool): Flag to gather mlm labels.
    - "mixed_precision" (bool): Casts input mask to fp16 if set to True.
      Otherwise, the generated mask is float32.
    """

    def __init__(self, params):
        super(BertCSVDynamicMaskDataProcessor, self).__init__()

        # Input params.
        self.meta_data = get_meta_data(params["data_dir"])

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0]
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.disable_nsp = params.get("disable_nsp", False)
        self.batch_size = params["batch_size"]

        self.num_batches = self.num_examples // self.batch_size
        assert (
            self.num_batches > 0
        ), "Dataset does not contain enough samples for one batch. Please choose a smaller batch size"
        self.num_tasks = get_world_size() if is_initialized() else 1
        self.num_batch_per_task = self.num_batches // self.num_tasks

        assert (
            self.num_batch_per_task > 0
        ), "Dataset cannot be evenly distributed across the given tasks. Please choose fewer tasks to run with"

        self.num_examples_per_task = self.num_batch_per_task * self.batch_size
        self.files_in_task = get_data_for_task(
            get_rank(),
            self.meta_data_values_cum_sum,
            self.num_examples_per_task,
            self.meta_data_values,
            self.meta_data_filenames,
        )

        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.repeat = params.get("repeat", False)
        self.mask_whole_word = params.get("mask_whole_word", False)
        self.do_lower = params.get("do_lower", False)
        self.dynamic_mlm_scale = params.get("dynamic_mlm_scale", False)
        self.buckets = params.get("buckets", None)

        # Multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        # Get special tokens and tokens that should not be masked.
        self.special_tokens = {
            "oov_token": params.get("oov_token", "[UNK]"),
            "mask_token": params.get("mask_token", "[MASK]"),
            "document_separator_token": params.get(
                "document_separator_token", "[SEP]"
            ),
        }
        self.exclude_from_masking = params.get(
            "exclude_from_masking", ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

        if self.do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }

            self.exclude_from_masking = list(
                map(lambda token: token.lower(), self.exclude_from_masking)
            )

        # Get vocab file and size.
        self.vocab_file = params["vocab_file"]
        self.vocab, self.vocab_size = build_vocab(
            self.vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )

        # Init tokenizer.
        self.tokenize = self.vocab.forward

        # Getting indices for special tokens.
        self.special_tokens_indices = {
            key: self.tokenize([value])[0]
            for key, value in self.special_tokens.items()
        }

        self.exclude_from_masking_ids = [
            self.tokenize([token])[0] for token in self.exclude_from_masking
        ]

        # We create a pool with tokens that can be used to randomly replace input tokens
        # for BERT MLM task.
        self.replacement_pool = list(
            set(range(self.vocab_size)) - set(self.exclude_from_masking_ids)
        )

        # Padding indices.
        # See https://huggingface.co/transformers/glossary.html#labels.
        self.labels_pad_id = params.get("labels_pad_id", 0)
        self.input_pad_id = params.get("input_pad_id", 0)
        self.attn_mask_pad_id = params.get("attn_mask_pad_id", 0)
        if not self.disable_nsp:
            self.segment_pad_id = params.get("segment_pad_id", 0)

        # Max sequence lengths size params.
        self.max_sequence_length = params["max_sequence_length"]
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
        self.gather_mlm_labels = params.get("gather_mlm_labels", True)

        # Store params.
        self.mp_type = (
            torch.float16 if params.get("mixed_precision") else torch.float32
        )
        self.data_buffer = []
        self.csv_files_per_task_per_worker = []
        self.processed_buffers = 0

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        if self.num_workers:
            dataloader = torch.utils.data.DataLoader(
                self,
                batch_size=None,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(self, batch_size=None)

        return dataloader

    def load_buffer(self):
        """
        Generator to read the data in chunks of size of `data_buffer`.

        :returns: Yields the data stored in the `data_buffer`.

        """

        self.data_buffer = []

        while self.processed_buffers < len(self.csv_files_per_task_per_worker):
            (
                current_file_path,
                num_examples,
                start_id,
            ) = self.csv_files_per_task_per_worker[self.processed_buffers]

            with open(current_file_path, "r", newline="") as fin:
                data_reader = csv.DictReader(fin)

                for row_id, row in enumerate(data_reader):
                    if start_id <= row_id < start_id + num_examples:
                        self.data_buffer.append(row)
                    else:
                        continue

                    if len(self.data_buffer) == self.shuffle_buffer:
                        if self.shuffle:
                            self.rng.shuffle(self.data_buffer)

                        for ind in range(len(self.data_buffer)):
                            yield self.data_buffer[ind]
                        self.data_buffer = []

                self.processed_buffers += 1

        if self.shuffle:
            self.rng.shuffle(self.data_buffer)

        for ind in range(len(self.data_buffer)):
            yield self.data_buffer[ind]
        self.data_buffer = []

    def __len__(self):
        # Returns the len of dataset on the task process
        if not self.drop_last:
            return (
                self.num_examples_per_task + self.batch_size - 1
            ) // self.batch_size
        elif self.buckets is None:
            return self.num_examples_per_task // self.batch_size
        else:
            # give an under-estimate in case we don't fully fill some buckets
            length = self.num_examples_per_task // self.batch_size
            length -= len(self.buckets)
            return length

    def get_single_item(self):
        """
        Iterating over the data to construct input features.

        :return: A tuple with training features:
            * np.array[int.32] input_ids: Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels.
               Shape: (`max_sequence_length`).
            * np.array[int.32] attention_mask
               Shape: (`max_sequence_length`).
            * np.array[int.32] token_type_ids: Numpy array with segment indices.
               Shape: (`max_sequence_length`).
            * np.array[int.32] next_sentence_label: Numpy array with labels for NSP task.
               Shape: (1).
            * np.array[int.32] masked_lm_mask: Numpy array with a mask of
               predicted tokens.
               Shape: (`max_predictions`)
               `0` indicates the non masked token, and `1` indicates the masked token.
        """
        # Shard the data across multiple processes.

        (
            self.processed_buffers,
            self.csv_files_per_task_per_worker,
            self.shuffle_seed,
            self.rng,
        ) = shard_and_shuffle_data(
            self.files_in_task, self.shuffle, self.shuffle_seed,
        )

        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            # `data_row` is a dict with keys:
            # ["tokens", "segment_ids", "is_random_next"].
            tokens = parse_text(data_row["tokens"], do_lower=self.do_lower)

            if self.disable_nsp:
                # truncate tokens to MSL
                tokens = tokens[: self.max_sequence_length]
            else:
                assert (
                    len(tokens) <= self.max_sequence_length
                ), "When using NSP head, make sure that len(tokens) <= MSL."

            (
                input_ids,
                labels,
                attention_mask,
                masked_lm_mask,
            ) = create_masked_lm_predictions(
                tokens,
                self.max_sequence_length,
                self.special_tokens_indices["mask_token"],
                self.max_predictions_per_seq,
                self.input_pad_id,
                self.attn_mask_pad_id,
                self.labels_pad_id,
                self.tokenize,
                self.vocab_size,
                self.masked_lm_prob,
                self.rng,
                self.exclude_from_masking,
                self.mask_whole_word,
                self.replacement_pool,
            )
            features = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if self.gather_mlm_labels:
                # Gather MLM positions
                _mlm_positions = np.nonzero(masked_lm_mask)[0]
                _num_preds = len(_mlm_positions)
                gathered_mlm_positions = np.zeros(
                    (self.max_predictions_per_seq,), dtype=np.int32
                )
                gathered_mlm_positions[:_num_preds] = _mlm_positions
                gathered_labels = np.zeros(
                    (self.max_predictions_per_seq,), dtype=np.int32
                )
                gathered_labels[:_num_preds] = labels[_mlm_positions]
                gathered_mlm_mask = np.zeros(
                    (self.max_predictions_per_seq,), dtype=np.int32
                )
                gathered_mlm_mask[:_num_preds] = masked_lm_mask[_mlm_positions]
                features["labels"] = gathered_labels
                features["masked_lm_mask"] = gathered_mlm_mask
                features["masked_lm_positions"] = gathered_mlm_positions
            else:
                features["labels"] = labels
                features["masked_lm_mask"] = masked_lm_mask

            if not self.disable_nsp:
                next_sentence_label = np.zeros((1,), dtype=np.int32)

                token_type_ids = (
                    np.ones((self.max_sequence_length,), dtype=np.int32)
                    * self.segment_pad_id
                )

                segment_ids = data_row["segment_ids"].strip("[]").split(", ")
                token_type_ids[: len(segment_ids)] = list(map(int, segment_ids))
                next_sentence_label[0] = int(data_row["is_random_next"])
                features["token_type_ids"] = token_type_ids
                features["next_sentence_label"] = next_sentence_label

            yield features

    def __iter__(self):
        batched_dataset = bucketed_batch(
            self.get_single_item(),
            self.batch_size,
            buckets=self.buckets,
            element_length_fn=lambda feats: np.sum(feats["attention_mask"]),
            drop_last=self.drop_last,
            seed=self.shuffle_seed,
        )
        for batch in batched_dataset:
            if self.dynamic_mlm_scale:
                scale = self.batch_size / torch.sum(batch["masked_lm_mask"])
                batch["mlm_loss_scale"] = scale.expand(self.batch_size, 1).to(
                    self.mp_type
                )
            yield batch


def train_input_dataloader(params):
    return getattr(
        sys.modules[__name__], params["train_input"]["data_processor"]
    )(params["train_input"]).create_dataloader(is_training=True)


def eval_input_dataloader(params):
    return getattr(
        sys.modules[__name__], params["eval_input"]["data_processor"]
    )(params["eval_input"]).create_dataloader(is_training=False)
