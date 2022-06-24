# LoL

This repository contains the code and trained models for our SIGIR paper [LoL: A Comparative Regularization Loss over Query Reformulation Losses for Pseudo-Relevance Feedback
](https://arxiv.org/abs/2204.11545).

## Usage

### Setup

Our experiments are conducted in the following environment with 4 V100 (32 GB) GPUs, where one GPU is dedicated to retrieval and the rest for reformulating queries.
At least two GPUs are needed to get it running.
```shell
conda create -n cr python=3.8
conda activate cr
conda install -c conda-forge openjdk=11 maven tensorboard jupyterlab ipywidgets
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=10.2 faiss-cpu -c pytorch
conda install -c huggingface -c conda-forge tokenizers=0.12.1 datasets=2.1.0 transformers=4.19.2
# If libs from huggingface don't work, try to install with pip
# pip install tokenizers==0.12.1 datasets==2.1.0 transformers==4.19.2
pip install scipy pyserini pytrec_eval
```

The corpus, datasets, document matrices, first-pass retrieval results, and model checkpoints can be downloaded from my [OneDrive](https://mailsucaseducn-my.sharepoint.com/:f:/g/personal/zhuyunchang17_mails_ucas_edu_cn/EkKyXIuEcDNAu2MpfRxdo_oB6fw8WdG4c3GUfVgKRfReeg).
After downloading, please merge them into this project.
Of these, the largest files are those matrices in `data/msmarco-passage/matrix/`.
If you don't want to download them, you need to generate them by running `notebooks/prepocess_index_*.ipynb`.
Simply put, those two notebooks convert the prebuilt-index loaded from [pyserini](https://github.com/castorini/pyserini) into a specified number of document vectors, which will be used during training or inference.

### For dense retrieval (ANCE)

#### Training

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RF=ance-bf
export SELECTION=top20
export EVAL_SELECTION=top100
export CKPT=ckpts/castorini/ance-msmarco-passage

# n_compare in [0, 1, 2], lambda_cr in [0, 0.5, 1, 1.5, 2]
export NC=1
python run.py \
    --corpus_path data/msmarco-passage/corpus.tsv \
    --pids_path data/msmarco-passage/pids.train.${RF}.${SELECTION}.txt \
    --matrix_path data/msmarco-passage/matrix/train.${RF}.${SELECTION}.pt \
    --eval_pids_path data/msmarco-passage/pids.dev.small.${RF}.${EVAL_SELECTION}.txt \
    --eval_matrix_path data/msmarco-passage/matrix/dev.small.${RF}.${EVAL_SELECTION}.pt \
    --queries_path data/msmarco-passage/queries.SPLIT.tsv \
    --qruns_path data/msmarco-passage/run/${RF}.SPLIT.tsv \
    --qrels_path data/msmarco-passage/qrels.SPLIT.txt \
    --prf --max_n_prf 5 --shuffle_psgs \
    --max_seq_len 512 --max_q_len 128 --max_p_len 128 \
    --dataloader_num_workers 36 --seed 42 \
    --model_type dqrd --model_name_or_path ${CKPT} --norm_power nan \
    --logging_dir runs --output_dir ckpts/${RF} --overwrite_output_dir \
    --do_train --fp16 --evaluation_strategy steps \
    --logging_steps 200 --eval_steps 1000 --save_steps 1000 \
    --metric_for_best_model mrr10 --greater_is_better True \
    --label_names poss_idx psgs_label \
    --n_compare ${NC} --num_train_epochs $((12/(1+${NC}))) \
    --per_device_train_batch_size $((36/(1+${NC}))) --per_device_eval_batch_size 36 \
    --warmup_ratio 0.1 --learning_rate 1e-5 --lambda_cr 1
```

#### Evaluation

```shell
# SPLIT: dev.small dl19-passage dl20-passage dlhard-passage
export RF=ance-bf
export CKPT=ckpts/ance-bf/dqrd_ance_lnan_pc_c1_b54_e6_1e-05_fp16/checkpoint-46000
export SPLIT=dev.small
python run.py \
    --corpus_path data/msmarco-passage/corpus.tsv \
    --full_pids_path data/msmarco-passage/pids.all.txt \
    --full_matrix_path data/msmarco-passage/matrix/${RF}.pt \
    --queries_path data/msmarco-passage/queries.${SPLIT}.tsv \
    --qruns_path data/msmarco-passage/run/${RF}.${SPLIT}.tsv \
    --qrels_path data/msmarco-passage/qrels.${SPLIT}.txt \
    --prf --max_n_prf 5 \
    --max_seq_len 512 --max_q_len 128 --max_p_len 128 \
    --dataloader_num_workers 24 \
    --model_type dqrd --model_name_or_path ${CKPT} --norm_power nan \
    --output_dir ${CKPT} \
    --do_predict \
    --per_device_eval_batch_size 24 \
    --label_names poss_idx psgs_label \
    --eval_board_path ckpts/${RF}/retr.${SPLIT}.prfK.metrics.tsv \
    --run_result_path ${CKPT}/retr.${SPLIT}.prfK.tsv

# Clean retrieval files
#rm ckpts/ance-bf/*/checkpoint-*/retr.dev.small.prf*.tsv
```

#### Prediction

```shell
export RF=ance-bf
export CKPT=ckpts/ance-bf/dqrd_ance_lnan_pc_c1_b54_e6_1e-05_fp16/checkpoint-46000
export SPLIT=eval.small
python run.py \
    --corpus_path data/msmarco-passage/corpus.tsv \
    --full_pids_path data/msmarco-passage/pids.all.txt \
    --full_matrix_path data/msmarco-passage/matrix/${RF}.pt \
    --queries_path data/msmarco-passage/queries.${SPLIT}.tsv \
    --qruns_path data/msmarco-passage/run/${RF}.${SPLIT}.tsv \
    --prf --max_n_prf 5 \
    --max_seq_len 512 --max_q_len 128 --max_p_len 128 \
    --dataloader_num_workers 24 \
    --model_type dqrd --model_name_or_path ${CKPT} --norm_power nan \
    --output_dir ${CKPT} \
    --do_predict \
    --per_device_eval_batch_size 24 \
    --label_names poss_idx psgs_label \
    --eval_board_path ckpts/${RF}/retr.${SPLIT}.prfK.metrics.tsv \
    --run_result_path ${CKPT}/retr.${SPLIT}.prfK.tsv

# Clean retrieval files
#rm ckpts/ance-bf/*/checkpoint-*/retr.eval.small.prf*.tsv
```

### For sparse retrieval (uniCOIL)

#### Training

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RF=unicoil-b8
export SELECTION=top10
export EVAL_SELECTION=top20
export CKPT=bert-base-uncased

export NC=1
python run.py \
    --corpus_path data/msmarco-passage/corpus.tsv \
    --pids_path data/msmarco-passage/pids.train.${RF}.${SELECTION}.txt \
    --matrix_path data/msmarco-passage/matrix/train.${RF}.${SELECTION}.coo.pt \
    --eval_pids_path data/msmarco-passage/pids.dev.small.${RF}.${EVAL_SELECTION}.txt \
    --eval_matrix_path data/msmarco-passage/matrix/dev.small.${RF}.${EVAL_SELECTION}.coo.pt \
    --queries_path data/msmarco-passage/queries.SPLIT.tsv \
    --qruns_path data/msmarco-passage/run/${RF}.SPLIT.tsv \
    --qrels_path data/msmarco-passage/qrels.SPLIT.txt \
    --prf --max_n_prf 5 --shuffle_psgs \
    --max_seq_len 512 --max_q_len 128 --max_p_len 128 \
    --dataloader_num_workers 36 --seed 42 \
    --model_type dqrsm --model_name_or_path ${CKPT} --norm_power 2 \
    --logging_dir runs --output_dir ckpts/${RF} --overwrite_output_dir \
    --do_train --fp16 --evaluation_strategy steps \
    --logging_steps 200 --eval_steps 1000 --save_steps 1000 \
    --metric_for_best_model mrr10 --greater_is_better True \
    --label_names poss_idx psgs_label \
    --n_compare ${NC} --num_train_epochs $((12/(1+${NC}))) \
    --per_device_train_batch_size $((36/(1+${NC}))) --per_device_eval_batch_size 36 \
    --warmup_ratio 0.1 --learning_rate 2e-5 --lambda_cr 1
```

#### Evaluation

```shell
# SPLIT: dev.small dl19-passage dl20-passage dlhard-passage
export RF=unicoil-b8
export CKPT=ckpts/unicoil-b8/dqrsm_bert_l2_pc_c1_b54_e6_2e-05_fp16/checkpoint-48000
export SPLIT=dev.small
python run.py \
    --corpus_path data/msmarco-passage/corpus.tsv \
    --full_pids_path data/msmarco-passage/pids.all.txt \
    --full_matrix_path data/msmarco-passage/matrix/${RF}.coo.pt \
    --queries_path data/msmarco-passage/queries.${SPLIT}.tsv \
    --qruns_path data/msmarco-passage/run/unicoil-b8.${SPLIT}.tsv \
    --qrels_path data/msmarco-passage/qrels.${SPLIT}.txt \
    --prf --max_n_prf 5 \
    --max_seq_len 512 --max_q_len 128 --max_p_len 128 \
    --dataloader_num_workers 24 \
    --model_type dqrsm --model_name_or_path ${CKPT} --norm_power 2 \
    --output_dir ${CKPT} \
    --do_predict \
    --per_device_eval_batch_size 24 \
    --label_names poss_idx psgs_label \
    --eval_board_path ckpts/${RF}/retr.${SPLIT}.prfK.metrics.tsv \
    --run_result_path ${CKPT}/retr.${SPLIT}.prfK.tsv

# Clean retrieval files
#rm ckpts/unicoil-b8/*/checkpoint-*/retr.dev.small.prf*.tsv
```

#### Prediction

```shell
export RF=unicoil-b8
export CKPT=ckpts/unicoil-b8/dqrsm_bert_l2_pc_c1_b54_e6_2e-05_fp16/checkpoint-48000
export SPLIT=eval.small
python run.py \
    --corpus_path data/msmarco-passage/corpus.tsv \
    --full_pids_path data/msmarco-passage/pids.all.txt \
    --full_matrix_path data/msmarco-passage/matrix/${RF}.coo.pt \
    --queries_path data/msmarco-passage/queries.${SPLIT}.tsv \
    --qruns_path data/msmarco-passage/run/unicoil-b8.${SPLIT}.tsv \
    --prf --max_n_prf 5 \
    --max_seq_len 512 --max_q_len 128 --max_p_len 128 \
    --dataloader_num_workers 24 \
    --model_type dqrsm --model_name_or_path ${CKPT} --norm_power 2 \
    --output_dir ${CKPT} \
    --do_predict \
    --per_device_eval_batch_size 24 \
    --label_names poss_idx psgs_label \
    --eval_board_path ckpts/${RF}/retr.${SPLIT}.prfK.metrics.tsv \
    --run_result_path ${CKPT}/retr.${SPLIT}.prfK.tsv

# Clean retrieval files
#rm ckpts/unicoil-b8/*/checkpoint-*/retr.eval.small.prf*.tsv
```


## Citation

If you use LoL in your work, please consider citing our paper:
```
@article{zhu2022lol,
  title={LoL: A Comparative Regularization Loss over Query Reformulation Losses for Pseudo-Relevance Feedback},
  author={Zhu, Yunchang and Pang, Liang and Lan, Yanyan and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2204.11545},
  year={2022}
}
```