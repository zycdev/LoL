import collections
import logging
import math
import os
import sys
import time
from types import MethodType
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING, Optional, Union
import pdb

from tqdm.auto import tqdm

# Integrations must be imported before ML frameworks:
from transformers.integrations import (
    hp_params,
)

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import DataCollator, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, __version__ as hg_ver
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_torch_tpu_available,
)
from transformers.integrations import TensorBoardCallback
from transformers.modeling_utils import unwrap_model
from transformers.trainer import Trainer, TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    denumpify_detensorize,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

if TYPE_CHECKING:
    import optuna

from arguments import DqrTrainingArguments
from modeling import DenseDqr
from util import f1_score, pairwise_hinge

logger = logging.getLogger(__name__)


def tb_on_train_begin(self, args, state, control, **kwargs):
    if not state.is_world_process_zero:
        return

    log_dir = None

    if state.is_hyper_param_search:
        trial_name = state.trial_name
        if trial_name is not None:
            log_dir = os.path.join(args.logging_dir, trial_name)

    if self.tb_writer is None:
        self._init_summary_writer(args, log_dir)

    if self.tb_writer is not None:
        self.tb_writer.add_text("args", args.to_json_string())
        if "model" in kwargs:
            model = kwargs["model"]
            if hasattr(model, "config") and model.config is not None:
                model_config_json = model.config.to_json_string()
                self.tb_writer.add_text("model_config", model_config_json)


class DqrTrainer(Trainer):

    def __init__(
            self,
            corpus_matrix: torch.Tensor,
            corpus_matrix_for_eval: Optional[torch.Tensor] = None,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: DqrTrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset,
                         tokenizer, model_init, compute_metrics, callbacks, optimizers)
        self.args: DqrTrainingArguments
        self.train_data_collator = train_data_collator if train_data_collator is not None else self.data_collator
        self.output_device = torch.device('cpu' if self.args.out_dev_idx < 0 else f'cuda:{self.args.out_dev_idx}')

        '''
        For sparse retrieval, there are two ways to retrieve in real time during training
        1. coo tensor in gpu for both search and loss computation
            - fast, but coo tensor must be small: place coo tensor in an additional gpu
        2. csr tensor in gpu (or csr matrix in cpu if sparse enough) for fast search, csr matrix in cpu for slicing,
           and sliced dense tensor moved to gpu for loss computation
            - slow due to frequent device conversion
        '''
        # (P, V)
        if corpus_matrix is not None:
            self.corpus_matrix = corpus_matrix.to(self.output_device)
            if not isinstance(self.model, DenseDqr):
                assert self.corpus_matrix.is_coalesced()
                df = torch.sparse.sum(self.corpus_matrix.bool().float(), dim=0).to_dense()
                self.idf = torch.log(self.corpus_matrix.size(0) / (1 + df))
            torch.cuda.empty_cache()
        else:
            self.corpus_matrix = None
            self.idf = None
        if corpus_matrix_for_eval is corpus_matrix:
            self.corpus_matrix_for_eval = self.corpus_matrix
        else:
            self.corpus_matrix_for_eval = None
            self.set_corpus_matrix_for_eval(corpus_matrix_for_eval)

        self.epoch_start_sr = float('inf')
        self.step_start_sr = float('inf')

        for cb in self.callback_handler.callbacks:
            if isinstance(cb, TensorBoardCallback):
                cb.on_train_begin = MethodType(tb_on_train_begin, cb)

    def set_corpus_matrix_for_eval(self, corpus_matrix):
        torch.cuda.empty_cache()
        if self.corpus_matrix_for_eval is not None:
            del self.corpus_matrix_for_eval
            self.corpus_matrix_for_eval = None
        if corpus_matrix is not None:
            self.corpus_matrix_for_eval = corpus_matrix.to(self.output_device)
            if not isinstance(self.model, DenseDqr):
                assert corpus_matrix.is_coalesced()
                # torch.sparse.mm not implemented for float16
                self.corpus_matrix_for_eval = self.corpus_matrix_for_eval.to_sparse_csr()
        torch.cuda.empty_cache()

    def _wrap_model(self, model, training=True):
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.args: DqrTrainingArguments
            model = nn.DataParallel(
                model,
                device_ids=list(range(self.args.out_dev_idx if self.args.n_gpu > 2 else self.args.n_gpu)),
                output_device=self.args.out_dev_idx
            )

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyway.
        if not training:
            return model

        if self.args.local_rank != -1:
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                find_unused_parameters = not model.is_gradient_checkpointing
            else:
                find_unused_parameters = True
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args.n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args.n_gpu != 0 else None,
                find_unused_parameters=find_unused_parameters,
            )

        return model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    @staticmethod
    def load_state_dict_to_module(state_dict_path, module, warning=True):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        load_result = module.load_state_dict(state_dict, strict=False)
        if not warning:
            return
        if len(load_result.missing_keys) != 0:
            logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    def _maybe_log_save_evaluate(self, tr_loss: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                 model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            if isinstance(tr_loss, dict) and 'loss' in tr_loss:
                for metric in tr_loss:
                    # all_gather + mean() to get average loss over all processes
                    metric_scalar = self._nested_gather(tr_loss[metric]).mean().item()
                    tr_loss[metric] -= tr_loss[metric]  # reset to zero
                    if metric == 'loss':
                        self._total_loss_scalar += metric_scalar
                    logs[metric] = round(metric_scalar / (self.state.global_step - self._globalstep_last_logged), 6)
            else:
                # all_gather + mean() to get average loss over all processes
                tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
                tr_loss -= tr_loss  # reset to zero
                self._total_loss_scalar += tr_loss_scalar
                logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 6)

            logs["learning_rate"] = self._get_learning_rate()

            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        if isinstance(train_dataset, IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.train_data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def infer(self, model, inputs, top_k=10, training=False):
        self.args: DqrTrainingArguments
        outputs: Tuple[torch.Tensor, torch.Tensor] = model(inputs['nn_input'], scoring=self.args.lambda_rr > 0)

        if self.args.lambda_pr <= 0:
            outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = outputs + (None,)
            return outputs, None

        # (Q, V)
        query_vector = outputs[0]
        # (P, V)
        corpus_matrix = self.corpus_matrix if training else self.corpus_matrix_for_eval
        # (P, Q)
        if self.args.fp16:
            with autocast(False):
                if corpus_matrix.is_sparse:
                    p_q_sim = torch.sparse.mm(corpus_matrix, query_vector.t().float())
                else:
                    p_q_sim = corpus_matrix.matmul(query_vector.t().float())
        else:
            if corpus_matrix.is_sparse:
                p_q_sim = torch.sparse.mm(corpus_matrix, query_vector.t())
            else:
                p_q_sim = corpus_matrix.matmul(query_vector.t())
        # (Q, top_k)
        top_psgs_idx = p_q_sim.topk(top_k, dim=0, sorted=True)[1].t()

        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = outputs + (top_psgs_idx,)

        return outputs, p_q_sim

    def compute_loss(self, model, inputs, return_outputs=False, training=True):
        top_k = 10 if training else 1000
        #        (P, Q)
        outputs, p_q_sim = self.infer(model, inputs, top_k, training)

        # (Q, V)      (Q, K)      (Q, top_k)
        query_vector, psgs_logit, top_psgs_idx = outputs

        poss_idx: List[List[int]] = inputs['poss_idx']  # (Q, _POS)
        psgs_label: List[List[float]] = inputs['psgs_label']  # (Q, _K)

        losses: Dict[str, torch.Tensor] = dict()

        self.args: DqrTrainingArguments
        if self.args.lambda_pr <= 0:
            assert psgs_logit is not None
            psgs_label_ = psgs_logit.new_tensor([y for _psgs_label in psgs_label for y in _psgs_label],
                                                dtype=torch.float32)
            psgs_logit_ = psgs_logit[~psgs_logit.isnan()]
            losses['loss_rr'] = F.binary_cross_entropy_with_logits(psgs_logit_, psgs_label_,
                                                                   reduction='mean', pos_weight=torch.tensor(7.0))
            losses['loss'] = losses['loss_rr']
            losses['f1'], precision, recall = f1_score(psgs_label_, (psgs_logit_ > 0).float())

            return (losses, outputs) if return_outputs else losses

        losses['mrr10'] = query_vector.new_zeros((), dtype=torch.float32)
        losses[f'recall{top_k}'] = query_vector.new_zeros((), dtype=torch.float32)
        for q, _poss_idx in enumerate(poss_idx):
            is_pos_ranked_top = top_psgs_idx[q] == _poss_idx[p_q_sim[_poss_idx, q].argmax()]  # (top_k,)
            if torch.any(is_pos_ranked_top):
                losses[f'recall{top_k}'] += 100.
                if torch.any(is_pos_ranked_top[:10]):
                    losses['mrr10'] += 100. / (is_pos_ranked_top[:10].nonzero()[0, 0] + 1.)
        losses['mrr10'] /= p_q_sim.size(1)
        losses[f'recall{top_k}'] /= p_q_sim.size(1)

        # query ranking
        pos_index_list: List[int] = []
        qry_index_list: List[int] = []
        pos_idx_to_queries_idx: Dict[int, List[int]] = dict()
        for q, _poss_idx in enumerate(poss_idx):
            if q % (1 + self.args.n_compare) == 0:
                pos_index_list.extend(_poss_idx)
                qry_index_list.extend([q] * len(_poss_idx))
            for pos_idx in _poss_idx:
                if pos_idx not in pos_idx_to_queries_idx:
                    pos_idx_to_queries_idx[pos_idx] = []
                pos_idx_to_queries_idx[pos_idx].append(q)
        pos_indices = p_q_sim.new_tensor(pos_index_list, dtype=torch.long)  # (|POS|,)
        q_ranking_label = p_q_sim.new_tensor(qry_index_list, dtype=torch.long)  # (|POS|,)
        pos_q_sim = torch.index_select(p_q_sim, dim=0, index=pos_indices)  # (|POS|, Q)
        for p, pos_idx in enumerate(pos_index_list):
            if len(pos_idx_to_queries_idx[pos_idx]) <= 1 + self.args.n_compare:
                continue
            for q in pos_idx_to_queries_idx[pos_idx]:
                if q // (1 + self.args.n_compare) != qry_index_list[p] // (1 + self.args.n_compare):
                    # mask other queries that have the same positive passage
                    pos_q_sim[p, q] = float('-inf')
        losses['loss_qr'] = F.cross_entropy(pos_q_sim, q_ranking_label, reduction='mean')

        # passage ranking
        q_p_sim = p_q_sim.t()  # (Q, P)
        # q_p_sim = torch.cat([p_q_sim, query_vector.sum(dim=1).unsqueeze(0)], dim=0).t()  # (Q, P + 1)
        p_ranking_label = p_q_sim.new_empty(p_q_sim.size(1), dtype=torch.long)  # (Q,)
        for q, _poss_idx in enumerate(poss_idx):
            p_ranking_label[q] = _poss_idx[q_p_sim[q, _poss_idx].argmin()]  # XXX argmin or random?
            for p in _poss_idx:
                if p != p_ranking_label[q]:  # mask other positive passages
                    q_p_sim[q, p] = float('-inf')
                else:
                    # q_p_sim[q, p] = q_p_sim[q, _poss_idx].mean()  # XXX min or mean
                    pass
        pr_loss = F.cross_entropy(q_p_sim, p_ranking_label, reduction='none')  # (Q,)
        losses['loss_pr'] = pr_loss.mean()

        # comparative regularization
        if self.args.n_compare > 0 and training:
            # (|original queries|, |revisions|)
            revision_effects = -pr_loss.view(-1, 1 + self.args.n_compare)
            effect_labels = torch.arange(revision_effects.size(1), 0, -1,
                                         device=revision_effects.device).unsqueeze(0).expand_as(revision_effects)
            losses['loss_cr'] = pairwise_hinge(revision_effects, effect_labels, margin=0., reduction='mean')

        # feedback-passage re-ranking
        if psgs_logit is not None:
            psgs_label_ = psgs_logit.new_tensor([y for _psgs_label in psgs_label for y in _psgs_label],
                                                dtype=torch.float32)
            psgs_logit_ = psgs_logit[~psgs_logit.isnan()]
            losses['loss_rr'] = F.binary_cross_entropy_with_logits(psgs_logit_, psgs_label_,
                                                                   reduction='mean', pos_weight=torch.tensor(7.0))
            losses['f1'], precision, recall = f1_score(psgs_label_, (psgs_logit_ > 0).float())

        # sparse regularization
        if self.args.lambda_sr > 0:
            losses['loss_sr'] = torch.linalg.norm(query_vector, ord=1, dim=1).mean()  # L1 regularization
            losses['loss_flops'] = torch.sum(query_vector.abs().mean(dim=0) ** 2)
            # XXX scheduler for lambda_sr
            if self.state.global_step <= self.step_start_sr:
                lambda_sr = 0
            else:
                ratio = math.ceil((self.state.global_step - self.step_start_sr) /
                                  (self.state.max_steps - self.step_start_sr) * 10) / 10
                lambda_sr = self.args.lambda_sr * ratio ** 2
            losses['lambda_sr'] = losses['loss_sr'].new_tensor(lambda_sr, dtype=torch.float32)
            losses['density'] = (query_vector != 0).sum(dim=1).float().mean()

        if self.args.fp16:
            with autocast(False):
                losses['loss'] = (self.args.lambda_pr * losses['loss_pr'].float() +
                                  self.args.lambda_qr * losses['loss_qr'].float())
                if 'loss_cr' in losses:
                    losses['loss'] += self.args.lambda_cr * losses['loss_cr'].float()
                if 'loss_rr' in losses:
                    losses['loss'] += self.args.lambda_rr * losses['loss_rr'].float()
                if 'loss_sr' in losses:
                    losses['loss'] += losses['lambda_sr'] * losses['loss_sr'].float()
        else:
            losses['loss'] = self.args.lambda_pr * losses['loss_pr'] + self.args.lambda_qr * losses['loss_qr']
            if 'loss_cr' in losses:
                losses['loss'] += self.args.lambda_cr * losses['loss_cr']
            if 'loss_rr' in losses:
                losses['loss'] += self.args.lambda_rr * losses['loss_rr']
            if 'loss_sr' in losses:
                losses['loss'] += losses['lambda_sr'] * losses['loss_sr']

        if losses['loss'].isnan() or losses['loss'].isinf():
            pdb.set_trace()
        if self.args.pdb and self.state.global_step % 100 == 0:
            pdb.set_trace()

        return (losses, outputs) if return_outputs else losses

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, torch.Tensor]:
        """Perform a training step on a batch of inputs.

        Args:
            model: The model to train.
            inputs: The inputs and targets of the model.

        Returns:
            Dict[str, torch.Tensor]: The Dict of tensors with detached training losses on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                losses = self.compute_loss(model, inputs)
        else:
            losses = self.compute_loss(model, inputs)
        loss = losses['loss']

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            losses['loss'] = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return {k: v.detach() for k, v in losses.items()}

    # noinspection PyAttributeOutsideInit
    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs,
    ):
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args: DqrTrainingArguments = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")
        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")
            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != hg_ver:
                    logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {hg_ver}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                self.load_state_dict_to_module(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), self.model.encoder)
                if os.path.exists(os.path.join(resume_from_checkpoint, 'model.pt')):
                    self.load_state_dict_to_module(os.path.join(resume_from_checkpoint, 'model.pt'), self.model, False)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size,
                # but it's the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs, so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in args.debug:
            if args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # each value in tr_losses is a tensor to avoid synchronization of TPUs through .item()
        tr_losses: Dict[str, torch.Tensor] = dict()
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # noinspection PyProtectedMember
                if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_losses_step = self.training_step(model, inputs)
                else:
                    tr_losses_step = self.training_step(model, inputs)

                for k, _loss in tr_losses_step.items():
                    if k not in tr_losses:
                        tr_losses[k] = torch.zeros_like(_loss)
                    if (args.logging_nan_inf_filter and not is_torch_tpu_available() and
                            (torch.isnan(_loss) or torch.isinf(_loss))):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_losses[k] += tr_losses[k] / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_losses[k] += _loss

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        (step + 1) == steps_in_epoch <= args.gradient_accumulation_steps
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_losses, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self.control.should_evaluate = args.do_eval
            self.control.should_save = args.do_eval
            self._maybe_log_save_evaluate(tr_losses, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here, so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            if os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)):
                self.load_state_dict_to_module(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME),
                                               self.model.encoder)
                if os.path.exists(os.path.join(self.state.best_model_checkpoint, 'model.pt')):
                    self.load_state_dict_to_module(os.path.join(self.state.best_model_checkpoint, 'model.pt'),
                                                   self.model, False)
            else:
                logger.warning(
                    f"Could not locate the best model at {self.state.best_model_checkpoint}, "
                    f"if you are running a distributed training on multiple nodes, "
                    f"you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_losses['loss'].item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.evaluation_loop(
            eval_dataloader, description="Evaluation", prediction_loss_only=True,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)
        self.args: DqrTrainingArguments
        if self.args.lambda_sr > 0 and self.step_start_sr >= self.state.global_step and \
                output.metrics[f'{metric_key_prefix}_mrr10'] > 34.0:  # XXX
            self.epoch_start_sr = int(self.state.epoch)
            self.step_start_sr = self.state.global_step
            logger.info(f"Start sparse regularization from E{self.epoch_start_sr}:S{self.step_start_sr}"
                        f"(mrr10={output.metrics[f'{metric_key_prefix}_mrr10']:.2f})")

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)

        output = self.evaluation_loop(
            test_dataloader, description="Prediction", prediction_loss_only=None,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if self.args.deepspeed and not self.deepspeed:
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        num_samples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {num_samples}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {dataloader.batch_size}")

        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        losses: Dict[str, torch.Tensor] = None
        retrieval_results: List[List[int]] = None
        observed_num_samples = 0
        for step, inputs in enumerate(dataloader):
            observed_batch_size = inputs['nn_input']['input_ids'].size(0)
            observed_num_samples += observed_batch_size

            if self.args.pdb and step % 100 == 0:
                pdb.set_trace()
            _losses, _outputs, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            _losses: Dict[str, torch.Tensor]
            _outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            if _losses is not None:
                if losses is None:
                    losses = _losses
                else:
                    for k, v in _losses.items():
                        losses[k] += v * observed_batch_size
            if _outputs is not None:
                # (Q, V)      (Q, K)      (Q, top_k)
                query_vector, psgs_logit, top_psgs_idx = _outputs
                if retrieval_results is None:
                    retrieval_results = []
                retrieval_results.extend(top_psgs_idx.tolist())

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
        assert num_samples == observed_num_samples

        if losses is not None:
            metrics = denumpify_detensorize({k: v / num_samples for k, v in losses.items()})
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = round(metrics.pop(key), 6)
        else:
            metrics = None

        return EvalLoopOutput(predictions=retrieval_results, label_ids=None, metrics=metrics, num_samples=num_samples)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if has_labels:
                labels = tuple(inputs.get(name) for name in self.label_names)
                if self.use_amp:
                    with autocast():
                        losses, outputs = self.compute_loss(model, inputs, return_outputs=True, training=False)
                else:
                    losses, outputs = self.compute_loss(model, inputs, return_outputs=True, training=False)
                losses = {k: v.mean().detach() for k, v in losses.items()}
            else:
                labels = None
                if self.use_amp:
                    with autocast():
                        outputs, _ = self.infer(model, inputs, top_k=1000, training=False)
                else:
                    outputs, _ = self.infer(model, inputs, top_k=1000, training=False)
                losses = None

        if prediction_loss_only:
            return losses, None, None

        return losses, outputs, labels
