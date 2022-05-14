from dataclasses import dataclass
import logging
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer

from util import pad_tensors

logger = logging.getLogger(__name__)


@dataclass
class QryRfmCollator:
    pad_token_id: int = 0
    n_compare: int = 0

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        if len(samples) == 0:
            return {}

        if self.n_compare > 0:
            samples_ = []
            for sample in samples:
                samples_.append(sample)
                for psg_num in sample['psg_nums'][1:]:
                    comparative_sample = sample.copy()
                    last_sep_index = comparative_sample['sep_indices'][psg_num]
                    comparative_sample['psgs_id'] = comparative_sample['psgs_id'][:psg_num]
                    comparative_sample['psgs_label'] = comparative_sample['psgs_label'][:psg_num]
                    comparative_sample['sep_indices'] = comparative_sample['sep_indices'][:psg_num + 1]
                    comparative_sample['input_ids'] = comparative_sample['input_ids'][:last_sep_index + 1]
                    comparative_sample['token_type_ids'] = comparative_sample['token_type_ids'][:last_sep_index + 1]
                    samples_.append(comparative_sample)
        else:
            samples_ = samples

        nn_input = {
            "input_ids": pad_tensors([sample['input_ids'] for sample in samples_], self.pad_token_id),  # (B, T)
            "attention_mask": pad_tensors([torch.ones_like(sample['input_ids']) for sample in samples_], 0),  # (B, T)
            "token_type_ids": pad_tensors([sample['token_type_ids'] for sample in samples_], 0),  # (B, T)
            "sep_indices": pad_tensors([sample['sep_indices'] for sample in samples_], -100),  # (B, 1 + K)
        }

        batch = {key: [sample[key] for sample in samples_] for key in samples_[0].keys() if key not in nn_input}
        batch['nn_input'] = nn_input

        return batch


class QryRfmDataset(Dataset):

    def __init__(
            self, tokenizer: PreTrainedTokenizer,
            corpus: Dict[str, str], pid2idx: Dict[str, int], queries: Dict[str, str],
            qruns: Optional[Dict[str, List[str]]] = None, qrels: Optional[Dict[str, Dict[str, int]]] = None,
            split: Optional[str] = 'test',
            prf: Optional[bool] = False, shuffle_psgs: Optional[bool] = False,
            n_compare: Optional[int] = 0,
            max_seq_len: Optional[int] = 512, max_q_len: Optional[int] = 128, max_p_len: Optional[int] = 128
    ):
        self.tokenizer = tokenizer

        self.corpus = corpus
        self.pid2idx = pid2idx
        self.queries = queries
        self.qruns = qruns
        self.qrels = qrels
        self.qrels_idx = None

        self.labeled = qrels is not None and len(qrels) > 0
        self.split = split
        if split == 'train':
            assert self.labeled
        self.prf = prf and qruns is not None and len(qruns) > 0
        self.shuffle_psgs = self.prf and shuffle_psgs
        self.n_compare = n_compare if self.prf else 0
        assert self.n_compare >= 0

        if self.labeled:
            self.qids: List[str] = sorted(qid for qid, rels in qrels.items() if sum(rels.values()) > 0)
            self.qrels_idx: Dict[str, List[int]] = {
                qid: [self.pid2idx[pid] for pid in qrels[qid] if qrels[qid][pid] > 0] for qid in self.qids
            }
        else:
            self.qids: List[str] = sorted(queries.keys())

        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len if self.prf else max_seq_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index) -> Dict[str, List]:
        qid = self.qids[index]
        qry = self.queries[qid]
        if not self.prf:
            top_pids = []
            psg_nums = [0]
        elif self.split != 'train':
            top_pids = self.qruns[qid]
            psg_nums = [len(self.qruns[qid])]
        else:
            psg_nums = sorted(random.sample(range(len(self.qruns[qid]) + 1), 1 + self.n_compare), reverse=True)
            max_n_psg = psg_nums[0]
            top_pids = random.sample(self.qruns[qid], max_n_psg) if self.shuffle_psgs else self.qruns[qid][:max_n_psg]

        sep_id = self.tokenizer.sep_token_id

        '''
        [CLS] q [SEP] p1 [SEP] p2 [SEP] p3 [SEP]
        '''
        input_ids: List[int] = self.tokenizer.encode(qry, add_special_tokens=True, truncation=True,
                                                     max_length=self.max_q_len + 2)
        token_type_ids = [0] * len(input_ids)
        sep_indices = [len(input_ids) - 1]
        psgs_id = []
        for i, pid in enumerate(top_pids):
            start = len(input_ids)
            if self.max_seq_len - start < 10:
                break
            input_ids += self.tokenizer.encode(self.corpus[pid], add_special_tokens=False, truncation=True,
                                               max_length=min(self.max_p_len, self.max_seq_len - start - 1)) + [sep_id]
            token_type_ids += [1] * (len(input_ids) - start)
            sep_indices.append(len(input_ids) - 1)
            psgs_id.append(pid)

        item = {
            "qid": qid,
            "psgs_id": psgs_id,
            "psg_nums": psg_nums,
            "sep_indices": torch.tensor(sep_indices, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        if not self.labeled:
            return item

        item['poss_idx'] = self.qrels_idx[qid]
        item['psgs_label'] = [float(pid in self.qrels[qid] and self.qrels[qid][pid] > 0) for pid in psgs_id]
        return item
