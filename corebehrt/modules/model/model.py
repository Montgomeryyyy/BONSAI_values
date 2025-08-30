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
    VAL_TOKEN,
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
        )
        self.is_causal = getattr(config, "is_causal", False)
        self.val_token_id = DEFAULT_VOCABULARY[VAL_TOKEN]

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
            segments=batch[SEGMENT_FEAT],
            age=batch[AGE_FEAT],
            abspos=batch[ABSPOS_FEAT],
            values=batch[VALUE_FEAT],
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
        
        # Handle value loss weight - make it learnable if specified
        value_loss_weight = getattr(config, "value_loss_weight", 1.0)
        learnable_value_weight = getattr(config, "learnable_value_weight", False)
        
        if learnable_value_weight:
            # Register as a learnable parameter
            self.register_parameter(
                "value_loss_weight", 
                nn.Parameter(torch.tensor(value_loss_weight, dtype=torch.float32))
            )
            logging.info(f"Value loss weight is learnable, initialized at {value_loss_weight}")
        else:
            # Keep as a fixed attribute
            self.value_loss_weight = value_loss_weight
            logging.info(f"Value loss weight is fixed at {value_loss_weight}")
            
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
        value_targets = batch.get(VALUE_FEAT)          # (B, L), values

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
            # Filter out [VAL] tokens from concept prediction
            # Only predict concepts for non-VAL positions
            non_val_mask = (labels != self.val_token_id)
            
            if non_val_mask.any():
                # Only compute loss for non-VAL tokens
                concept_logits = logits[non_val_mask]
                concept_labels = labels[non_val_mask]
                concept_loss = self.get_loss(concept_logits, concept_labels)
            else:
                concept_loss = torch.tensor(0.0, device=last_hidden_state.device)
            
            outputs.concept_loss = concept_loss
            outputs.labels = labels
        else:
            concept_loss = torch.tensor(0.0, device=last_hidden_state.device)

        # === Value Prediction ===
        # Only predict values for positions that have VAL tokens
        is_val_token = (labels == self.val_token_id) if labels is not None else None
        
        value_loss = torch.tensor(0.0, device=last_hidden_state.device)
        if value_targets is not None and is_val_token is not None:
            if self.sparse_prediction:
                # In sparse mode, we already have the masked positions
                # Check which of these are VAL tokens
                val_positions = is_val_token
                if val_positions.any():
                    # Get the corresponding value targets for VAL positions
                    val_targets = value_targets[val_positions]
                    val_mask = ~torch.isnan(val_targets)  # Remove NaN values
                    
                    if val_mask.any():
                        # Predict values only for [VAL] tokens with valid targets
                        val_hidden = last_hidden_state[val_positions][val_mask]
                        predicted_values = self.val_head(val_hidden).squeeze(-1)
                        target_values = val_targets[val_mask]
                        
                        value_loss = self.val_loss_fct(predicted_values, target_values)
                        outputs.predicted_values = predicted_values
            else:
                # Not sparse: use full shape (B, L)
                # Find [VAL] token positions in the full sequence
                val_positions = (labels == self.val_token_id)
                if val_positions.any():
                    val_targets = value_targets[val_positions]
                    val_mask = ~torch.isnan(val_targets)
                    
                    if val_mask.any():
                        val_hidden = last_hidden_state[val_positions][val_mask]
                        predicted_values = self.val_head(val_hidden).squeeze(-1)
                        target_values = val_targets[val_mask]
                        
                        value_loss = self.val_loss_fct(predicted_values, target_values)
                        outputs.predicted_values = predicted_values

            outputs.value_loss = value_loss

        # === Final loss ===
        if labels is not None and value_targets is not None:
            # Handle both learnable parameter and fixed value cases
            if hasattr(self, 'value_loss_weight') and isinstance(self.value_loss_weight, nn.Parameter):
                weight = self.value_loss_weight
            else:
                weight = self.value_loss_weight
            outputs.loss = concept_loss + weight * value_loss
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
