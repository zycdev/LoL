# coding=utf-8
# Copyright 2022 LoL authors
# Copyright 2021 COIL authors
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
XXX:
    norm_power: 2 or 1
    log_saturation: w/o > w.
    max_p_len: 128 or 192
    shuffle_psgs: w. > w/o
"""
from datetime import datetime
import logging
import os
import re
import sys

import pandas as pd
import torch

from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser, set_seed
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME

from arguments import DataArguments, ModelArguments, DqrTrainingArguments
from trainer import DqrTrainer
from modeling import SparseDqr, SparseMlmDqr, SparseSeqDqr, DenseDqr
from qr_dataset import QryRfmDataset, QryRfmCollator
from util import load_corpus, load_pids, load_queries, load_qruns, load_qrels, compute_metrics

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "dqrs": SparseDqr,
    "dqrsm": SparseMlmDqr,
    "dqrss": SparseSeqDqr,
    "dqrd": DenseDqr
}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DqrTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, it's the path to a json file
        model_args, data_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        train_args: DqrTrainingArguments

    if train_args.do_train:
        if 'unicoil' in model_args.model_name_or_path.replace('/unicoil-b8', ''):
            init_model = 'unicoil'
        elif 'ance' in model_args.model_name_or_path.replace('/ance-bf', ''):
            init_model = 'ance'
        else:
            assert 'bert' in model_args.model_name_or_path
            init_model = 'bert'
        loss_key = ''.join(['p' if train_args.lambda_pr > 0 else '',
                            'q' if train_args.lambda_qr > 0 else '',
                            'c' if train_args.lambda_cr > 0 else '',
                            'r' if train_args.lambda_rr > 0 else '',
                            's' if train_args.lambda_sr > 0 else ''])
        model_brief = (
            f"{model_args.model_type}_{init_model}_l{model_args.norm_power:.0f}_{'lgs_' if model_args.log_sat else ''}"
            f"{loss_key}_c{train_args.n_compare}_b{train_args.train_batch_size}_"
            f"e{train_args.num_train_epochs:.0f}_{train_args.learning_rate}{'_fp16' if train_args.fp16 else ''}"
        )
        if train_args.comment:
            model_brief += f".{train_args.comment}"
        train_args.run_name = model_brief
        train_args.output_dir = os.path.join(train_args.output_dir, model_brief)
        train_args.logging_dir = os.path.join(train_args.logging_dir,
                                              f"{datetime.now().strftime('%m%d%H%M')}_{model_brief}")
        if (os.path.exists(train_args.output_dir) and os.listdir(train_args.output_dir) and
                not train_args.overwrite_output_dir):
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                f"Use --overwrite_output_dir to overcome."
            )
        if (
                model_args.model_name_or_path.startswith(train_args.output_dir) and
                os.path.isdir(model_args.model_name_or_path) and
                os.path.isfile(os.path.join(model_args.model_name_or_path, OPTIMIZER_NAME)) and
                os.path.isfile(os.path.join(model_args.model_name_or_path, SCHEDULER_NAME))
        ):
            train_args.resume_from_checkpoint = model_args.model_name_or_path
        else:
            train_args.resume_from_checkpoint = None

    if train_args.do_predict and os.path.exists(data_args.run_result_path):
        if all(os.path.exists(data_args.run_result_path.replace('prfK', f'prf{n_prf}'))
               for n_prf in range(data_args.max_n_prf + 1)):
            raise FileExistsError(f'result files {data_args.run_result_path} already exist')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation arguments {train_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    # Set seed
    set_seed(train_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = MODEL_CLASSES[model_args.model_type].from_pretrained(
        data_args, model_args, train_args,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    corpus = load_corpus(data_args.corpus_path)
    logger.info(f"Loaded corpus ({len(corpus)}) from {data_args.corpus_path}")

    # Get datasets
    if train_args.do_train:
        pid2idx_for_train = load_pids(data_args.pids_path)[1]
        logger.info(f"Loaded {len(pid2idx_for_train)} passages ids for training from {data_args.pids_path}")
        corpus_matrix_for_train = torch.load(data_args.matrix_path, map_location='cpu')
        if model_args.model_type != 'dqrd':
            corpus_matrix_for_train = corpus_matrix_for_train.coalesce()
        logger.info(f"Loaded matrix {corpus_matrix_for_train.shape} of corpus "
                    f"for training from {data_args.matrix_path}")
        assert len(pid2idx_for_train) == corpus_matrix_for_train.shape[0]

        logger.info(f"Loading train set")
        train_queries = load_queries(data_args.queries_path.replace('SPLIT', 'train'))
        train_qruns = load_qruns(data_args.qruns_path.replace('SPLIT', 'train'), data_args.max_n_prf)
        train_qrels = load_qrels(data_args.qrels_path.replace('SPLIT', 'train'))
        train_dataset = QryRfmDataset(
            tokenizer, corpus, pid2idx_for_train, train_queries, train_qruns, train_qrels,
            split='train', prf=data_args.prf, shuffle_psgs=data_args.shuffle_psgs,
            n_compare=train_args.n_compare, max_seq_len=data_args.max_seq_len,
            max_q_len=data_args.max_q_len, max_p_len=data_args.max_p_len
        )
    else:
        train_dataset = None
        corpus_matrix_for_train = None

    if train_args.do_eval:
        pid2idx_for_dev = load_pids(data_args.eval_pids_path)[1]
        logger.info(f"Loaded {len(pid2idx_for_dev)} passages ids for evaluation from {data_args.eval_pids_path}")
        if data_args.eval_matrix_path == data_args.matrix_path:
            assert corpus_matrix_for_train is not None
            corpus_matrix_for_dev = corpus_matrix_for_train
        else:
            corpus_matrix_for_dev = torch.load(data_args.eval_matrix_path, map_location='cpu')
            if model_args.model_type != 'dqrd':
                corpus_matrix_for_dev = corpus_matrix_for_dev.coalesce()
        logger.info(f"Loaded matrix {corpus_matrix_for_dev.shape} of corpus "
                    f"for evaluation from {data_args.eval_matrix_path}")
        assert len(pid2idx_for_dev) == corpus_matrix_for_dev.shape[0]

        logger.info(f"Loading dev set")
        dev_queries = load_queries(data_args.queries_path.replace('SPLIT', 'dev.small'))
        dev_qruns = load_qruns(data_args.qruns_path.replace('SPLIT', 'dev.small'), data_args.max_n_prf)
        dev_qrels = load_qrels(data_args.qrels_path.replace('SPLIT', 'dev.small'))
        dev_dataset = QryRfmDataset(
            tokenizer, corpus, pid2idx_for_dev, dev_queries, dev_qruns, dev_qrels,
            split='dev', prf=data_args.prf, shuffle_psgs=False, n_compare=0,
            max_seq_len=data_args.max_seq_len, max_q_len=data_args.max_q_len, max_p_len=data_args.max_p_len
        )
    else:
        dev_dataset = None
        corpus_matrix_for_dev = None

    # Initialize our Trainer
    trainer = DqrTrainer(
        corpus_matrix=corpus_matrix_for_train,
        corpus_matrix_for_eval=corpus_matrix_for_dev,
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=QryRfmCollator(pad_token_id=tokenizer.pad_token_id, n_compare=0),
        train_data_collator=QryRfmCollator(pad_token_id=tokenizer.pad_token_id, n_compare=train_args.n_compare)
    )
    del corpus_matrix_for_train, corpus_matrix_for_dev

    if train_args.do_train:
        trainer.train(train_args.resume_from_checkpoint)

    if train_args.do_predict:
        result_dir = os.path.split(data_args.run_result_path)[0]
        if not os.path.exists(result_dir):
            logger.info(f'Creating result directory {result_dir}')
            os.makedirs(result_dir)

        pids, pid2idx = load_pids(data_args.full_pids_path)
        logger.info(f"Loaded {len(pid2idx)} passages ids for prediction from {data_args.full_pids_path}")
        corpus_matrix = torch.load(data_args.full_matrix_path, map_location='cpu')
        if model_args.model_type != 'dqrd':
            corpus_matrix = corpus_matrix.coalesce()
        logger.info(f"Loaded matrix {corpus_matrix.shape} of corpus for prediction from {data_args.full_matrix_path}")
        assert len(pid2idx) == corpus_matrix.shape[0]
        trainer.set_corpus_matrix_for_eval(corpus_matrix)
        del corpus_matrix

        test_queries = load_queries(data_args.queries_path)
        if data_args.qrels_path is not None and os.path.exists(data_args.qrels_path):
            qrels = load_qrels(data_args.qrels_path)
        else:
            qrels = None
        if '.dl' in data_args.queries_path:
            metrics_to_sort = ["NDCG@10", "Recall@1000"]
        else:
            metrics_to_sort = ["MRR@10", "Recall@1000"]
        metrics_to_save = ["Recall@10", "Recall@1000", "MRR@10", "MRR@1000",
                           "MAP@10", "MAP@1000", "NDCG@10", "NDCG@100"]
        for n_prf in range(data_args.max_n_prf + 1):
            logger.info(f"Predicting for PRF{n_prf}")
            run_result_path = data_args.run_result_path.replace('prfK', f'prf{n_prf}')
            if os.path.exists(run_result_path):
                continue

            test_qruns = load_qruns(data_args.qruns_path, n_prf) if n_prf > 0 else None
            test_dataset = QryRfmDataset(
                tokenizer, corpus, pid2idx, test_queries, test_qruns,
                split='test', prf=data_args.prf and n_prf > 0, shuffle_psgs=False, n_compare=0,
                max_seq_len=data_args.max_seq_len, max_q_len=data_args.max_q_len, max_p_len=data_args.max_p_len
            )

            retrieval_results = trainer.predict(test_dataset=test_dataset).predictions

            if trainer.is_world_process_zero():
                assert len(test_dataset.qids) == len(retrieval_results)
                qp_scores = dict()
                with open(run_result_path, 'w') as f:
                    for qid, sorted_indices in zip(test_dataset.qids, retrieval_results):
                        qp_scores[qid] = dict()
                        for rank, pidx in enumerate(sorted_indices):
                            pid = pids[pidx]
                            qp_scores[qid][pid] = len(sorted_indices) - rank
                            f.write(f'{qid}\t{pid}\t{rank + 1}\n')

                if qrels is not None:
                    metrics = compute_metrics(qrels, qp_scores, [10, 100, 1000], data_args.rel_level,
                                              data_args.eval_result_path.replace('prfK', f'prf{n_prf}'))
                    prfs_evals_path = data_args.eval_result_path.replace('.prfK', '').replace('.json', '.tsv')
                    if os.path.exists(prfs_evals_path):
                        prfs_metrics = pd.read_csv(prfs_evals_path, sep='\t', index_col=0).to_dict(orient='index')
                    else:
                        prfs_metrics = dict()
                    prfs_metrics[n_prf] = metrics
                    df = pd.DataFrame.from_dict(prfs_metrics, orient='index', columns=metrics_to_save)
                    df.to_csv(prfs_evals_path, sep='\t', float_format='%.3f')

                    if re.search(r'checkpoint-(\d+)', model_args.model_name_or_path):
                        ckpt_step = int(re.search(r'checkpoint-(\d+)', model_args.model_name_or_path).group(1))
                    else:
                        ckpt_step = 'final'
                    ckpts_evals_path = data_args.eval_results_path.replace('prfK', f'prf{n_prf}')
                    if os.path.exists(ckpts_evals_path):
                        # shutil.copy(ckpts_evals_path, f'{ckpts_evals_path}.bak')
                        ckpts_metrics = pd.read_csv(ckpts_evals_path, sep='\t', index_col=0).to_dict(orient='index')
                    else:
                        ckpts_metrics = dict()
                    ckpts_metrics[ckpt_step] = metrics
                    df = pd.DataFrame.from_dict(ckpts_metrics, orient='index', columns=metrics_to_save)
                    df.sort_values(by=metrics_to_sort, ascending=False, inplace=True)
                    df.to_csv(ckpts_evals_path, sep='\t', float_format='%.3f')

                    model_brief = os.path.split(re.sub(r'(/checkpoint-\d+)?/?$', '', model_args.model_name_or_path))[1]
                    models_evals_path = data_args.eval_board_path.replace('prfK', f'prf{n_prf}')
                    if os.path.exists(models_evals_path):
                        # shutil.copy(models_evals_path, f'{models_evals_path}.bak')
                        models_metrics = pd.read_csv(models_evals_path, sep='\t', index_col=0).to_dict(orient='index')
                    else:
                        models_metrics = dict()
                    if (
                            model_brief not in models_metrics or
                            tuple(df.iloc[0][metrics_to_sort]) > tuple(models_metrics[model_brief][k]
                                                                       for k in metrics_to_sort)
                    ):
                        models_metrics[model_brief] = df.iloc[0].to_dict()
                    df = pd.DataFrame.from_dict(models_metrics, orient='index')
                    df.sort_values(by=metrics_to_sort, ascending=False, inplace=True)
                    df.to_csv(models_evals_path, sep='\t', float_format='%.3f')


if __name__ == "__main__":
    main()
