from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
import re
from typing import Optional

from transformers import TrainingArguments
from transformers.file_utils import ExplicitEnum, torch_required

logger = logging.getLogger(__name__)


class ModelType(ExplicitEnum):
    DQRS = "dqrs"
    DQRS_MLM = "dqrsm"
    DQRS_TW = "dqrss"
    DQRD = "dqrd"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: ModelType = field(
        metadata={"help": "The model class to use"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    norm_power: float = field(default=2)

    # for sparse
    log_sat: bool = field(default=False, metadata={"help": "Whether to use log-saturation on vector weights"})
    no_relu: bool = field(default=False)

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DataArguments:
    corpus_path: str = field(default=None, metadata={"help": "Path to the tsv file of corpus"})
    pids_path: str = field(default=None, metadata={"help": "Path to the file of indexed pids"})
    matrix_path: str = field(default=None, metadata={"help": "Path to the coo matrix of corpus"})
    eval_pids_path: str = field(default=None, metadata={"help": "Path to the file of indexed pids for evaluation"})
    eval_matrix_path: str = field(default=None, metadata={"help": "Path to the coo matrix of corpus for evaluation"})
    full_pids_path: str = field(default=None, metadata={"help": "Path to the file of all pids"})
    full_matrix_path: str = field(default=None, metadata={"help": "Path to the coo matrix of full corpus"})
    queries_path: str = field(default=None, metadata={"help": "Path to the tsv file of queries"})
    qruns_path: str = field(default=None, metadata={"help": "Path to the tsv file of retrieval results"})
    qrels_path: str = field(default=None, metadata={"help": "Path to the tsv file of retrieval labels"})

    prf: bool = field(default=False)
    max_n_prf: int = field(default=5)
    shuffle_psgs: bool = field(default=False)

    max_seq_len: int = field(default=512)
    max_q_len: int = field(default=128)
    max_p_len: int = field(default=128)

    rel_level: int = field(default=1)
    run_result_path: str = field(default=None, metadata={"help": "where to save the rank result"})
    eval_result_path: str = field(default=None, metadata={"help": "where to save the evaluation metrics"})
    eval_results_path: str = field(default=None, metadata={"help": "where to save evaluation metrics of checkpoints"})
    eval_board_path: str = field(default=None, metadata={"help": "where to save the evaluation metrics of all runs"})

    def __post_init__(self):
        if '.dl' in self.queries_path:
            self.rel_level = 2
        if self.run_result_path is not None and self.eval_result_path is None:
            self.eval_result_path = f"{self.run_result_path.rsplit('.', 1)[0]}.metric.json"
        if self.eval_result_path is not None and self.eval_results_path is None:
            self.eval_results_path = re.sub(r'checkpoint-\d+/', '', self.eval_result_path).replace('metric.json',
                                                                                                   'metrics.tsv')
        if not self.prf:
            self.max_n_prf = 0

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DqrTrainingArguments(TrainingArguments):
    logging_nan_inf_filter: bool = field(default=False, metadata={"help": "Filter nan and inf losses for logging."})

    n_compare: int = field(default=0, metadata={"help": "The number of comparative samples for each query."})
    lambda_pr: float = field(default=1.0, metadata={"help": "The weight of passage ranking loss."})
    lambda_qr: float = field(default=0.0, metadata={"help": "The weight of query ranking loss."})
    lambda_rr: float = field(default=0.0, metadata={"help": "The weight of re-ranking loss."})
    lambda_cr: float = field(default=0.0, metadata={"help": "The weight of comparative regularization."})
    lambda_sr: float = field(default=0.0, metadata={"help": "The weight of sparse regularization."})

    comment: str = field(default=None)

    pdb: bool = field(default=False)

    do_encode: bool = field(default=False, metadata={"help": "Whether to run encoding on the test set."})

    def __post_init__(self):
        super().__post_init__()
        self.n_compare = max(0, self.n_compare)
        self.lambda_pr = max(0.0, self.lambda_pr)
        self.lambda_qr = max(0.0, self.lambda_qr) if self.lambda_pr > 0 else 0.0
        self.lambda_cr = max(0.0, self.lambda_cr) if self.n_compare > 0 and self.lambda_pr > 0 else 0.0
        self.lambda_rr = max(0.0, self.lambda_rr)
        self.lambda_sr = max(0.0, self.lambda_sr) if self.lambda_pr > 0 else 0.0
        if not self.do_train:
            self.lambda_qr = 0.0
            self.lambda_cr = 0.0
            self.lambda_rr = 0.0
            self.lambda_sr = 0.0

    @property
    @torch_required
    def out_dev_idx(self) -> int:
        return self.n_gpu - 1

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        if self.n_gpu <= 1:
            train_batch_size = per_device_batch_size
        else:
            train_batch_size = per_device_batch_size * max(2, self.out_dev_idx)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        if self.n_gpu <= 1:
            eval_batch_size = per_device_batch_size
        else:
            eval_batch_size = per_device_batch_size * max(2, self.out_dev_idx)
        return eval_batch_size
