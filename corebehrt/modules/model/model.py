"""
Module: corebehrt_module

This module defines customized EHR-focused BERT models built on top of ModernBertModel:

- CorebehrtEncoder: replaces token embeddings with temporal EHR embeddings and causal encoder layers.
- CorebehrtForPretraining: extends the encoder for masked language model pretraining on EHR sequences.
- CorebehrtForFineTuning: extends the encoder for downstream classification/regression tasks on EHR data.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import ModernBertModel
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead

from corebehrt.constants.data import (
    ABSPOS_FEAT,
    AGE_FEAT,
    ATTENTION_MASK,
    CONCEPT_FEAT,
    DEFAULT_VOCABULARY,
    PAD_TOKEN,
    SEGMENT_FEAT,
    TARGET,
    VALUE_FEAT,
    VALUE_NULL_TOKEN,
)
from corebehrt.constants.model import (
    TIME2VEC_ABSPOS_SCALE,
    TIME2VEC_ABSPOS_SHIFT,
    TIME2VEC_AGE_SCALE,
    TIME2VEC_AGE_SHIFT,
)
from corebehrt.functional.modeling.attention import make_attention_causal
from corebehrt.modules.model.embeddings import EhrEmbeddings
from corebehrt.modules.model.heads import FineTuneHead

logger = logging.getLogger(__name__)


class CorebehrtEncoder(ModernBertModel):
    """
    Encoder backbone for EHR data using ModernBert.

    Attributes:
        embeddings (EhrEmbeddings): custom embeddings for concepts, segments, age, and absolute position.
        layers (nn.ModuleList): list of causal encoder layers replacing standard BERT layers.
    """

    def __init__(self, config):
        super().__init__(config)
        # config.is_decoder = True
        # config.add_cross_attention = False
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
            pad_token_id=config.pad_token_id,
            age_scale=getattr(config, "age_scale", TIME2VEC_AGE_SCALE),
            age_shift=getattr(config, "age_shift", TIME2VEC_AGE_SHIFT),
            abspos_scale=getattr(config, "abspos_scale", TIME2VEC_ABSPOS_SCALE),
            abspos_shift=getattr(config, "abspos_shift", TIME2VEC_ABSPOS_SHIFT),
            value_null_token=getattr(config, "value_null_token", VALUE_NULL_TOKEN),
            value_embedding_mode=getattr(config, "value_embedding_mode", "concat")
        )
        self.is_causal = getattr(config, "is_causal", False)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass building embeddings and attention mask, then calling ModernBertModel.

        Args:
            batch (dict): must contain:
                - "concept": Tensor of token indices (B, L)
                - "segment": Tensor of segment IDs (B, L)
                - "age": Tensor of patient ages (B, L)
                - "abspos": Tensor of absolute position values (B, L)
            **kwargs: Additional arguments to pass to the ModernBertModel forward method

        Returns:
            BaseModelOutput: output of ModernBertModel with last_hidden_state, etc.
        """
        if ATTENTION_MASK in batch:
            attention_mask = batch[ATTENTION_MASK]
        else:
            attention_mask = (
                batch[CONCEPT_FEAT] != DEFAULT_VOCABULARY[PAD_TOKEN]
            ).float()

        inputs_embeds = self.embeddings(
            input_ids=batch[CONCEPT_FEAT],
            values=batch[VALUE_FEAT],
            segments=batch[SEGMENT_FEAT],
            age=batch[AGE_FEAT],
            abspos=batch[ABSPOS_FEAT],
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _update_attention_mask(
        self, attention_mask: torch.Tensor, output_attentions: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls super()._update_attention_mask and adds causal masking if self.is_causal=True.
        Returns:
            Tuple of:
            - Global attention mask
            - Sliding window mask for local attention
        """
        global_attention_mask, sliding_window_mask = super()._update_attention_mask(
            attention_mask, output_attentions
        )
        if self.is_causal:
            global_attention_mask = make_attention_causal(global_attention_mask)
            sliding_window_mask = make_attention_causal(sliding_window_mask)

        return global_attention_mask, sliding_window_mask


class CorebehrtForPretraining(CorebehrtEncoder):
    """
    Masked Language Model head for EHR pretraining.

    Adds a prediction head and linear decoder on top of CorebehrtEncoder.
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.head = ModernBertPredictionHead(config)
        self.val_head = nn.Linear(config.hidden_size, 1)
        self.val_loss_fct = nn.MSELoss()
        self.value_loss_weight = getattr(config, "value_loss_weight", 1.0)
        self.value_null_token = getattr(config, "value_null_token", VALUE_NULL_TOKEN)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

    # Inspiration from ModernBertForMaskedLM
    def forward(self, batch: dict, **kwargs):
        outputs = super().forward(batch, **kwargs)
        last_hidden_state = outputs[0]  # (B, L, H)

        # === Concept Prediction ===
        labels = batch.get(TARGET)                     # (B, L), masked tokens only
        value_targets = batch.get(VALUE_FEAT)          # (B, L), values or VALUE_NULL_TOKEN

        if self.sparse_prediction and labels is not None:
            # Flatten and filter non-masked tokens
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        # Predict concepts
        logits = self.decoder(self.head(last_hidden_state))  # (N, vocab_size)
        outputs.logits = logits

        if labels is not None:
            concept_loss = self.get_loss(logits, labels)
            outputs.concept_loss = concept_loss
            outputs.labels = labels
        else:
            concept_loss = torch.tensor(0.0, device=last_hidden_state.device)

        # === Value Prediction ===
        predicted_values = self.val_head(last_hidden_state).squeeze(-1)  # (N,) if flattened

        value_loss = torch.tensor(0.0, device=last_hidden_state.device)
        if value_targets is not None:
            if self.sparse_prediction:
                value_targets = value_targets.view(-1)
                value_targets = value_targets[mask_tokens]  # same mask as concepts
                value_mask = (value_targets != self.value_null_token)

                if value_mask.any():
                    value_loss = self.val_loss_fct(
                        predicted_values[value_mask], value_targets[value_mask]
                    )
            else:
                # Not sparse: use full shape (B, L)
                pred_vals = self.val_head(last_hidden_state).squeeze(-1)  # (B, L)
                value_mask = (value_targets != self.value_null_token)
                pred_vals = pred_vals[value_mask]
                target_vals = value_targets[value_mask]

                if target_vals.numel() > 0:
                    value_loss = self.val_loss_fct(pred_vals, target_vals)

            outputs.value_loss = value_loss
            outputs.predicted_values = predicted_values

        # === Final loss ===
        if labels is not None and value_targets is not None:
            outputs.loss = concept_loss + self.value_loss_weight * value_loss
        else:
            outputs.loss = concept_loss

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class CorebehrtForFineTuning(CorebehrtEncoder):
    """
    Fine-tuning head for downstream classification on EHR sequences.

    Adds a binary classification head (BCEWithLogits) on top of sequence outputs.
    """

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "pos_weight", None):
            pos_weight = torch.tensor(config.pos_weight)
        else:
            pos_weight = None

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict, **kwargs):
        """
        Forward pass for fine-tuning.

        Args:
            batch (dict): must contain 'concept', 'segment', 'age', 'abspos', 'attention_mask';
                          optional 'target' as labels.
            **kwargs: Additional arguments to pass to the encoder forward method

        Returns:
            BaseModelOutput: with logits and optional loss if target provided.
        """
        outputs = super().forward(batch, **kwargs)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch[ATTENTION_MASK])
        outputs.logits = logits

        if batch.get(TARGET) is not None:
            outputs.loss = self.get_loss(logits, batch[TARGET])

        return outputs

    def get_loss(self, hidden_states, labels):
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))
