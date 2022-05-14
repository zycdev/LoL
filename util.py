import json
import logging
from itertools import product
import re
from typing import Dict, List, Tuple, Iterable, Optional, Union

import torch
import torch.nn.functional as F

import pytrec_eval

logger = logging.getLogger(__name__)


def compute_hole(qrels: Dict[str, Dict[str, int]],
                 qp_scores: Dict[str, Dict[str, float]],
                 k_values: List[int]) -> Dict[str, Dict[str, float]]:
    metrics = {f"Hole@{k}": dict() for k in sorted(k_values)}

    annotated_corpus = set()
    for _, rels in qrels.items():
        for pid, rel in rels.items():
            annotated_corpus.add(pid)

    k_max = max(k_values)
    for qid, pid2score in qp_scores.items():
        top_hits = sorted(pid2score.items(), key=lambda item: item[1], reverse=True)[:k_max]
        for k in k_values:
            hole_psgs = [hit[0] for hit in top_hits[0:k] if hit[0] not in annotated_corpus]
            metrics[f"Hole@{k}"][qid] = 100.0 * len(hole_psgs) / k

    return metrics


def compute_mrr(qrels: Dict[str, Dict[str, int]],
                qp_scores: Dict[str, Dict[str, float]],
                k_values: List[int]) -> Dict[str, Dict[str, float]]:
    metrics = {f"MRR@{k}": dict() for k in sorted(k_values)}

    k_max, top_hits = max(k_values), dict()
    for qid, pid2score in qp_scores.items():
        top_hits[qid] = sorted(pid2score.items(), key=lambda item: item[1], reverse=True)[:k_max]
    for qid in qrels:
        rel_pids = set(pid for pid, rel in qrels[qid].items() if rel > 0)
        for k in k_values:
            for rank, hit in enumerate(top_hits[qid][0:k]):
                if hit[0] in rel_pids:
                    metrics[f"MRR@{k}"][qid] = 100.0 / (rank + 1)
                    break
            if qid not in metrics[f"MRR@{k}"]:
                metrics[f"MRR@{k}"][qid] = 0.0

    return metrics


def compute_metrics(qrels: Dict[str, Dict[str, int]], qp_scores: Dict[str, Dict[str, float]],
                    k_values: List[int] = (10, 1000), relevance_level=1, save_path=None) -> Dict[str, float]:
    all_metrics = compute_mrr(qrels, qp_scores, k_values)
    all_metrics.update(compute_hole(qrels, qp_scores, k_values))
    for measure in ['MAP', 'NDCG', 'Recall', 'P']:
        for k in sorted(k_values):
            if f'{measure}@{k}' not in all_metrics:
                all_metrics[f'{measure}@{k}'] = dict()

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    measures = {map_string, ndcg_string, recall_string, precision_string}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures, relevance_level)
    scores = evaluator.evaluate(qp_scores)
    for qid in scores.keys():
        for k in k_values:
            all_metrics[f"MAP@{k}"][qid] = scores[qid][f"map_cut_{k}"] * 100.
            all_metrics[f"NDCG@{k}"][qid] = scores[qid][f"ndcg_cut_{k}"] * 100.
            all_metrics[f"Recall@{k}"][qid] = scores[qid][f"recall_{k}"] * 100.
            all_metrics[f"P@{k}"][qid] = scores[qid][f"P_{k}"] * 100.

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(all_metrics, f)
    avg_metrics = {measure: round(sum(results.values()) / len(qrels), 3) for measure, results in all_metrics.items()}
    for measure, result in avg_metrics.items():
        logger.info(f"{measure}: {result:.2f}")

    return avg_metrics


def f1_score(y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon) * 100.
    recall = tp / (tp + fn + epsilon) * 100.
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1, precision, recall


def load_queries(qry_tsv: str) -> Dict[str, str]:
    queries = dict()
    pattern = re.compile(r'_{3,}')
    with open(qry_tsv) as f:
        for line in f:
            qid, qry = line.split('\t')
            queries[qid.strip()] = pattern.sub('___', qry.strip())
    return queries


def load_pids(pid_txt: str) -> Tuple[List[str], Dict[str, int]]:
    pids = []
    pid2idx = dict()
    with open(pid_txt) as f:
        for line in f:
            pid = line.strip()
            pid2idx[pid] = len(pids)
            pids.append(pid)
    return pids, pid2idx


def load_corpus(corpus_tsv: str, pids: Optional[Iterable] = None) -> Dict[str, str]:
    corpus = dict()
    with open(corpus_tsv) as f:
        for line in f:
            pid, psg = line.split('\t')
            pid, psg = pid.strip(), psg.strip()
            if pids is None or pid in pids:
                corpus[pid] = psg
    return corpus


def load_qrels(qrel_tsv: str) -> Dict[str, Dict[str, int]]:
    qrels = dict()
    with open(qrel_tsv) as f:
        for line in f:
            qid, _, pid, rel = line.split()
            # qid, pid, rel = qid.strip(), pid.strip(), rel.strip()
            if qid not in qrels:
                qrels[qid] = dict()
            qrels[qid][pid] = int(rel)
    return qrels  # {k: list(v.keys()) for k, v in qrels.items()}


def load_qruns(qrun_tsv: str, top_k: Optional[int] = 5) -> Union[Dict[str, List[str]], None]:
    if top_k < 1:
        return None
    qruns = dict()
    with open(qrun_tsv) as f:
        for line in f:
            qid, pid, rank = line.split('\t')
            qid, pid, rank = qid.strip(), pid.strip(), int(rank)
            if rank > top_k:
                continue
            if qid not in qruns:
                qruns[qid] = []
            qruns[qid].append(pid)
    return qruns


def pad_tensors(tensors: List[torch.Tensor], pad_val, left_pad=False, move_eos_to_beginning=False, eos_val=None):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_val
            dst[0] = eos_val
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if len(tensors[0].size()) > 1:
        tensors = [x.view(-1) for x in tensors]
    batch_size = len(tensors)
    max_len = max(x.size(0) for x in tensors)
    padded_tensor = tensors[0].new_full((batch_size, max_len), pad_val, requires_grad=tensors[0].requires_grad)
    for i, x in enumerate(tensors):
        copy_tensor(x, padded_tensor[i, max_len - len(x):] if left_pad else padded_tensor[i, :len(x)])
    return padded_tensor


def list_mle(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None,
             reduction: Optional[str] = 'mean', eps: Optional[float] = 1e-10) -> torch.Tensor:
    """ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".

    Args:
        y_pred: (N, L) predictions from the model
        y_true: (N, L) ground truth labels
        mask: (N, L) 1 for available position, 0 for masked position
        reduction: 'none' | 'mean' | 'sum'
        eps: epsilon value, used for numerical stability
    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    # shuffle for randomized tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    shuffled_y_pred = y_pred[:, random_indices]
    shuffled_y_true = y_true[:, random_indices]
    shuffled_mask = mask[:, random_indices] if mask is not None else None

    sorted_y_true, rank_true = shuffled_y_true.sort(descending=True, dim=1)
    y_pred_in_true_order = shuffled_y_pred.gather(dim=1, index=rank_true)
    if shuffled_mask is not None:
        y_pred_in_true_order = y_pred_in_true_order - 10000.0 * (1.0 - shuffled_mask)

    max_y_pred, _ = y_pred_in_true_order.max(dim=1, keepdim=True)
    y_pred_in_true_order_minus_max = y_pred_in_true_order - max_y_pred
    cum_sum = y_pred_in_true_order_minus_max.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    observation_loss = torch.log(cum_sum + eps) - y_pred_in_true_order_minus_max
    if shuffled_mask is not None:
        observation_loss[shuffled_mask == 0] = 0.0
    loss = observation_loss.sum(dim=1)

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def pairwise_hinge(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   margin: Optional[float] = 0., reduction: Optional[str] = 'mean') -> torch.Tensor:
    """RankNet loss introduced in "Learning to Rank using Gradient Descent".

    Args:
        y_pred: (N, L) predictions from the model
        y_true: (N, L) ground truth labels
        mask: (N, L) 1 for available position, 0 for masked position
        margin:
        reduction: 'none' | 'mean' | 'sum'
    Returns:
        torch.Tensor: scalar if `reduction` is not 'none' else (N,)
    """
    if mask is not None:
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        y_pred[mask == 0] = float('-inf')
        y_true[mask == 0] = float('-inf')

    # generate every pair of indices from the range of candidates number in the batch
    candidate_pairs = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, candidate_pairs]
    pairs_pred = y_pred[:, candidate_pairs]

    # calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]

    # filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symmetric pairs, so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    s1 = pairs_pred[:, :, 0][the_mask]
    s2 = pairs_pred[:, :, 1][the_mask]  # .detach()  # XXX
    target = the_mask.float()[the_mask]

    return F.margin_ranking_loss(s1, s2, target, margin=margin, reduction=reduction)
