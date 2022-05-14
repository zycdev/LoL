import math
import os
import logging
from typing import Dict, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForMaskedLM, BertModel, BertForMaskedLM, RobertaModel
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from arguments import ModelArguments, DataArguments, DqrTrainingArguments

logger = logging.getLogger(__name__)


class Dqr(nn.Module):
    """
    Differentiable Query Reformulation
    """

    def __init__(self, encoder: Union[BertModel, BertForMaskedLM, RobertaModel],
                 data_args: DataArguments, model_args: ModelArguments, train_args: DqrTrainingArguments):
        super(Dqr, self).__init__()
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args
        self.encoder = encoder
        self.psg_transform = BertPredictionHeadTransform(self.encoder.config)
        self.reranker = nn.Linear(self.encoder.config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, data_args: DataArguments, model_args: ModelArguments, train_args: DqrTrainingArguments,
                        *args, **kwargs):
        model_path = model_args.model_name_or_path
        if cls is SparseMlmDqr:
            encoder = AutoModelForMaskedLM.from_pretrained(model_path, *args, **kwargs)
        else:
            encoder = AutoModel.from_pretrained(model_path, *args, **kwargs)
        model = cls(encoder, data_args, model_args, train_args)
        if os.path.exists(os.path.join(model_path, 'model.pt')):
            logger.info(f"loading extra weights from {os.path.join(model_path, 'model.pt')}")
            model_dict = torch.load(os.path.join(model_path, 'model.pt'), map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        model_dict = self.state_dict()
        encoder_parameter_keys = [k for k in model_dict.keys() if k.startswith('encoder')]
        for k in encoder_parameter_keys:
            model_dict.pop(k)
        if len(model_dict) > 0:
            torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

    def score_psgs(self, hidden_vectors: torch.FloatTensor, sep_indices: torch.LongTensor) -> torch.FloatTensor:
        # hidden_vectors: (B, T, H), sep_indices: (B, K + 1)
        psgs_mask = sep_indices.new_zeros((sep_indices.size(0), sep_indices.size(1) - 1), dtype=torch.bool)  # (B, K)
        psg_vectors = []  # (P, H)
        for q, _sep_indices in enumerate(sep_indices):
            for p, sep_idx in enumerate(_sep_indices[:-1]):
                next_sep_idx = _sep_indices[p + 1]
                if next_sep_idx <= sep_idx + 1:
                    break
                psgs_mask[q, p] = 1
                psg_vectors.append(hidden_vectors[q, sep_idx + 1:next_sep_idx].mean(dim=0))

        if len(psg_vectors) > 0:
            psg_logits = []  # (P,)
            psg_idx = 0
            chunk_size = len(psg_vectors)  # hidden_vectors.size(0) * 3
            while psg_idx < len(psg_vectors):
                chunk = torch.stack(psg_vectors[psg_idx:psg_idx + chunk_size], dim=0)
                psg_logits.append(self.reranker(self.psg_transform(chunk)).view(-1))
                psg_idx += len(psg_logits[-1])
            psg_logits = torch.cat(psg_logits, dim=0)
            psgs_logit = torch.full_like(psgs_mask, float('nan'), dtype=psg_logits.dtype)  # (B, K)
            psgs_logit[psgs_mask] = psg_logits
        else:
            psgs_logit = torch.full_like(psgs_mask, float('nan'), dtype=hidden_vectors.dtype)  # (B, K)

        return psgs_logit

    def forward(self, batch: Dict, scoring=True):
        # (B, T, H)
        if isinstance(self.encoder, RobertaModel):
            hidden_vectors = self.encoder(batch['input_ids'], batch['attention_mask'])[0]
        else:
            hidden_vectors = self.encoder(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])[0]
        # (B, P)
        if scoring:
            psgs_logit = self.score_psgs(hidden_vectors, batch['sep_indices'])
        else:
            psgs_logit = None

        return psgs_logit, hidden_vectors


class SparseMlmDqr(Dqr):
    def __init__(self, encoder: BertForMaskedLM,
                 data_args: DataArguments, model_args: ModelArguments, train_args: DqrTrainingArguments):
        super(Dqr, self).__init__()
        self.data_args, self.model_args, self.train_args = data_args, model_args, train_args
        self.encoder = encoder
        # self.encoder.bert.embeddings.word_embeddings.weight.requires_grad = False
        self.psg_transform = BertPredictionHeadTransform(self.encoder.config)
        self.reranker = nn.Linear(self.encoder.config.hidden_size, 1)
        # 0: [PAD], 100: [UNK], 101: [CLS], 102: [SEP], 103: [MASK]
        self.special_token_ids = list(i for i in range(999) if i not in [100])

    @staticmethod
    def score_psgs2(hidden_vectors: torch.FloatTensor, sep_indices: torch.LongTensor) -> torch.FloatTensor:
        # hidden_vectors: (B, T, H), sep_indices: (B, K + 1)
        # (B, K)
        psgs_logit = hidden_vectors.new_full((sep_indices.size(0), sep_indices.size(1) - 1), float('nan'))
        for q, _sep_indices in enumerate(sep_indices):
            qry_vector = None  # (H,)
            psg_vectors = []  # (P, H)
            for p, sep_idx in enumerate(_sep_indices[:-1]):
                if p == 0:
                    qry_vector = hidden_vectors[q, 1:sep_idx].mean(dim=0)
                next_sep_idx = _sep_indices[p + 1]
                if next_sep_idx <= sep_idx + 1:
                    break
                psg_vectors.append(hidden_vectors[q, sep_idx + 1:next_sep_idx].mean(dim=0))
            if len(psg_vectors) > 0:
                psg_vectors = torch.stack(psg_vectors, dim=0)
                psg_logits = torch.matmul(psg_vectors, qry_vector)  # (P,)
                psgs_logit[q, :psg_logits.size(0)] = psg_logits

        return psgs_logit

    def forward(self, batch: Dict, scoring=True):
        # (B, T, H)
        hidden_vectors = self.encoder.bert(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])[0]

        # (B, K)
        if scoring:
            psgs_logit = self.score_psgs(hidden_vectors, batch['sep_indices'])
        else:
            psgs_logit = None

        attention_mask = batch['attention_mask'].bool()  # (B, T)
        real_sparse_vectors = self.encoder.cls(hidden_vectors[attention_mask])  # (_, V)
        # (B, T, V)
        sparse_vectors = real_sparse_vectors.new_full(hidden_vectors.shape[:2] + real_sparse_vectors.shape[1:], -100.)
        sparse_vectors[attention_mask] = real_sparse_vectors

        # (B, V)
        sparse_vector = sparse_vectors.max(dim=1).values
        sparse_vector[:, self.special_token_ids] = 0.
        if not self.model_args.no_relu:
            sparse_vector.relu_()
        if self.model_args.log_sat:
            sparse_vector = torch.log(1 + sparse_vector)
        sparse_vector = F.normalize(sparse_vector, p=self.model_args.norm_power, dim=1)

        return sparse_vector, psgs_logit


class SparseSeqDqr(Dqr):
    def __init__(self, encoder: BertModel,
                 data_args: DataArguments, model_args: ModelArguments, train_args: DqrTrainingArguments):
        super(SparseSeqDqr, self).__init__(encoder, data_args, model_args, train_args)
        self.tok_proj = nn.Linear(self.encoder.config.hidden_size, 1)
        # 0: [PAD], 100: [UNK], 101: [CLS], 102: [SEP], 103: [MASK]
        self.special_token_ids = list(i for i in range(999) if i not in [100])

    def forward(self, batch: Dict, scoring=True):
        # (B, P)    (B, T, H)
        psgs_logit, hidden_vectors = super().forward(batch, scoring)

        # (B, T)
        tok_weights = self.tok_proj(hidden_vectors).squeeze() * batch['attention_mask']
        if not self.model_args.no_relu:
            tok_weights.relu_()

        # (B, T, V)
        sparse_vectors = tok_weights.new_zeros(tuple(tok_weights.size()) + (self.encoder.config.vocab_size,))
        sparse_vectors.scatter_(dim=2, index=batch['input_ids'].unsqueeze(-1), src=tok_weights.unsqueeze(-1))
        # (B, V)
        sparse_vector = sparse_vectors.max(dim=1).values
        if self.model_args.log_sat:
            sparse_vector = torch.log(1 + sparse_vector)

        sparse_vector[:, self.special_token_ids] = 0.
        sparse_vector.renorm_(p=self.model_args.norm_power, dim=0, maxnorm=1)
        # sparse_vector = F.normalize(sparse_vector, p=self.model_args.norm_power, dim=1)

        return sparse_vector, psgs_logit


class SparseDqr(Dqr):
    def __init__(self, encoder: BertModel,
                 data_args: DataArguments, model_args: ModelArguments, train_args: DqrTrainingArguments):
        super(SparseDqr, self).__init__(encoder, data_args, model_args, train_args)
        self.sparse_proj = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.vocab_size, bias=True)
        self.sparse_proj.weight = self.encoder.embeddings.word_embeddings.weight
        # self.sparse_proj.weight.requires_grad = False
        # 0: [PAD], 100: [UNK], 101: [CLS], 102: [SEP], 103: [MASK]
        self.special_token_ids = list(i for i in range(999) if i not in [100])

    def forward(self, batch: Dict, scoring=True):
        # (B, P)    (B, T, H)
        psgs_logit, hidden_vectors = super().forward(batch, scoring)

        # (B, T, V)
        sparse_vectors = self.sparse_proj(hidden_vectors) * batch['attention_mask'].unsqueeze(-1)

        # (B, V)
        sparse_vector = sparse_vectors.max(dim=1).values
        sparse_vector[:, self.special_token_ids] = 0.
        if not self.model_args.no_relu:
            sparse_vector.relu_()
        if self.model_args.log_sat:
            sparse_vector = torch.log(1 + sparse_vector)
        sparse_vector = F.normalize(sparse_vector, p=self.model_args.norm_power, dim=1)

        return sparse_vector, psgs_logit


class DenseDqr(Dqr):
    def __init__(self, encoder: BertModel,
                 data_args: DataArguments, model_args: ModelArguments, train_args: DqrTrainingArguments):
        super(DenseDqr, self).__init__(encoder, data_args, model_args, train_args)
        self.dense_proj = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(self.encoder.config.hidden_size)

    def forward(self, batch: Dict, scoring=True):
        # (B, P)    (B, T, H)
        psgs_logit, hidden_vectors = super().forward(batch, scoring)

        # (B, H)
        dense_vector = self.dense_proj(hidden_vectors[:, 0])

        dense_vector = self.layer_norm(dense_vector)
        if not math.isnan(self.model_args.norm_power):
            dense_vector = F.normalize(dense_vector, p=self.model_args.norm_power, dim=1)

        return dense_vector, psgs_logit
