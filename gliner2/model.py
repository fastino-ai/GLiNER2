import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gliner.modeling.span_rep import SpanRepLayer
from gliner2.layers import CountLSTMoE, CountLSTM, create_mlp, CountLSTMv2
from gliner2.processor import SchemaTransformer
from gliner2.infer_packing import (
    InferencePackingConfig,
    pack_requests,
    unpack_spans,
)
from gliner2.packing_utils import (
    apply_pair_attention_encoder,
    build_position_ids,
)
from safetensors.torch import save_file, load_file
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)


# ---------------------------
# ExtractorConfig Definition
# ---------------------------
class ExtractorConfig(PretrainedConfig):
    model_type = "extractor"

    def __init__(
            self,
            model_name="bert-base-uncased",
            max_width=8,
            counting_layer="count_lstm",
            token_pooling="first",
            pack_sequences=False,
            pack_max_length=None,
            pack_streams_per_batch=1,
            **kwargs
    ):
        """
        Configuration for the Extractor model.

        Args:
            model_name (str): Pretrained model name for the encoder.
            max_width (int): Maximum width for span representations.
            counting_layer (str): Type of counting layer to use.
            token_pooling (str): Method for pooling tokens (default: "first").
            pack_sequences (bool): Enable sequence packing for encoder inputs.
            pack_max_length (Optional[int]): Max length for packed streams (defaults to encoder limit).
            pack_streams_per_batch (int): Number of packed streams per batch.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_width = max_width
        self.counting_layer = counting_layer
        self.token_pooling = token_pooling
        self.pack_sequences = pack_sequences
        self.pack_max_length = pack_max_length
        self.pack_streams_per_batch = pack_streams_per_batch

# ---------------------------
# Extractor Model
# ---------------------------
class Extractor(PreTrainedModel):
    config_class = ExtractorConfig

    def __init__(self, config, encoder_config=None, tokenizer=None):
        """
        Initializes the Extractor model.

        Args:
            config (ExtractorConfig): Model configuration.
            encoder_config (Optional[dict or PretrainedConfig]): Configuration for the encoder.
            tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for the processor.
        """
        super().__init__(config)
        self.config = config
        self.max_width = config.max_width

        # Initialize the SchemaTransformer. If a tokenizer is provided, use it.
        if tokenizer is not None:
            self.processor = SchemaTransformer(tokenizer=tokenizer, token_pooling=config.token_pooling)
        else:
            self.processor = SchemaTransformer(config.model_name, token_pooling=config.token_pooling)
        self._inference_packing_config: Optional[InferencePackingConfig] = None

        # Load the encoder model.
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config, trust_remote_code=True)
        else:
            self.encoder = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
        # Resize token embeddings using the processor's tokenizer.
        self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.hidden_size = self.encoder.config.hidden_size

        # Span representation layer.
        self.span_rep = SpanRepLayer(
            span_mode="markerV0",
            hidden_size=self.hidden_size,
            max_width=self.max_width,
            dropout=0.1,
        )
        # Classifier for classification tasks.
        self.classifier = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=1,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )
        # Layers for hierarchical schema count prediction.
        self.count_pred = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=20,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )
        # Count embedding module.
        if config.counting_layer == "count_lstm":
            self.count_embed = CountLSTM(self.hidden_size)
        elif config.counting_layer == "count_lstm_moe":
            self.count_embed = CountLSTMoE(
                hidden_size=self.hidden_size,
                n_experts=4,
                ffn_mult=2,
                dropout=0.1
            )
        elif config.counting_layer == "count_lstm_v2":
            self.count_embed = CountLSTMv2(hidden_size=self.hidden_size)

        # print information about the model
        print("=" * 60)
        print("🧠  Model Configuration")
        print("=" * 60)
        print(f"Encoder model      : {config.model_name}")
        print(f"Counting layer     : {config.counting_layer}")
        print(f"Token pooling      : {config.token_pooling}")
        print("=" * 60)

    # ---------------------------
    # Packing Helpers
    # ---------------------------
    def configure_inference_packing(self, config: Optional[InferencePackingConfig]) -> None:
        """
        Configure default packing behaviour for inference calls.

        Passing ``None`` disables packing by default. Individual inference
        methods can override via a packing_config argument.
        """

        self._inference_packing_config = config

    def _build_packing_config(self, override: Optional[InferencePackingConfig] = None) -> Optional[InferencePackingConfig]:
        if override is not None:
            return override
        if self._inference_packing_config is not None:
            return self._inference_packing_config
        if not getattr(self.config, "pack_sequences", False):
            return None

        max_len = self.config.pack_max_length
        if max_len is None:
            max_len = getattr(self.encoder.config, "max_position_embeddings", None)
        if max_len is None:
            return None

        sep_id = getattr(self.processor.tokenizer, "sep_token_id", None)
        if sep_id is None:
            sep_id = getattr(self.processor.tokenizer, "eos_token_id", None)

        return InferencePackingConfig(
            max_length=int(max_len),
            sep_token_id=sep_id,
            streams_per_batch=getattr(self.config, "pack_streams_per_batch", 1),
        )

    def _encode_with_optional_packing(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            packing_config: Optional[InferencePackingConfig] = None,
            **kwargs,
    ):
        """
        Run the encoder with optional packing. Returns an object exposing
        .last_hidden_state, mirroring Hugging Face outputs.
        """
        if packing_config is None:
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        lengths = attention_mask.sum(dim=-1).tolist()
        seq_len = int(input_ids.size(1))
        if not lengths or all(int(l) == seq_len for l in lengths):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        requests = []
        for row, length in zip(input_ids, lengths):
            length = int(length)
            if length <= 0:
                requests.append({"input_ids": []})
            else:
                requests.append({"input_ids": row[:length].tolist()})

        pad_token_id = getattr(self.encoder.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", 0)

        packed = pack_requests(requests, packing_config, pad_token_id)

        device = input_ids.device
        packed_ids = packed.input_ids.to(device=device)
        pair_mask = packed.pair_attention_mask.to(device=device)
        fallback_mask = packed.attention_mask.to(device=device)
        position_ids = build_position_ids(packed)

        outputs = apply_pair_attention_encoder(
            self.encoder,
            packed_ids,
            pair_mask,
            fallback_mask,
            position_ids=position_ids,
            **kwargs,
        )

        unpacked: List[torch.Tensor] = unpack_spans(outputs.last_hidden_state, packed)
        hidden_size = outputs.last_hidden_state.size(-1)
        batch, seq = input_ids.size()
        padded_out = outputs.last_hidden_state.new_zeros(batch, seq, hidden_size)
        for idx, target in enumerate(unpacked):
            tgt_len = int(target.size(0))
            if tgt_len == 0:
                continue
            end = min(tgt_len, seq)
            padded_out[idx, :end] = target[:end]

        class _Result:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return _Result(padded_out)

    # ---------------------------
    # Original Methods
    # ---------------------------
    def compute_span_rep(self, token_embeddings: torch.Tensor) -> dict:
        text_length = len(token_embeddings)
        device = next(self.encoder.parameters()).device

        spans_idx = []
        for i in range(text_length):
            for j in range(self.max_width):
                if i + j < text_length:
                    spans_idx.append((i, i + j))
                else:
                    spans_idx.append((-1, -1))  # Explicitly mark invalid spans

        spans_idx = torch.LongTensor([spans_idx]).to(device)  # shape: [1, num_spans, 2]

        # Mask invalid spans (where either start or end == -1)
        span_mask = (spans_idx[:, :, 0] == -1) | (spans_idx[:, :, 1] == -1)
        span_mask = span_mask.to(device)

        # Replace invalid spans with dummy (0, 0) for safe indexing
        safe_spans_idx = torch.where(
            span_mask.unsqueeze(-1),
            torch.zeros_like(spans_idx),
            spans_idx
        )

        # Compute span representations (batch size = 1)
        span_rep = self.span_rep(token_embeddings.unsqueeze(0), safe_spans_idx).squeeze(0)  # [num_spans, hidden_dim]

        return {
            "span_rep": span_rep,
            "spans_idx": spans_idx,
            "span_mask": span_mask,
        }

    def classification_loss(
            self,
            embs_per_schema: List[List[torch.Tensor]],
            task_types: List[str],
            structure_labels: List[Any]
    ) -> torch.Tensor:
        """
        Helper to compute classification loss for a single sample.

        Parameters
        ----------
        embs_per_schema : List[List[torch.Tensor]]
            Embeddings per schema for this sample
        task_types : List[str]
            Task types for each schema
        structure_labels : List[Any]
            Structure labels including classification labels

        Returns
        -------
        torch.Tensor
            Classification loss for this sample
        """
        cls_embeds = []
        binary_labels = []

        for i, task_type in enumerate(task_types):
            if task_type == "classifications":
                # Handle empty embeddings
                if not embs_per_schema[i]:
                    continue

                schema_embs = torch.stack(embs_per_schema[i], dim=0)
                cls_embeds.append(schema_embs[1:])  # Skip [P] token
                binary_labels.extend(structure_labels[i])

        if not cls_embeds:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        cls_embeds = torch.cat(cls_embeds, dim=0)
        logits = self.classifier(cls_embeds)
        binary_labels = torch.FloatTensor(binary_labels).to(logits.device)

        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        return loss_fn(logits.squeeze(-1), binary_labels)

    def compute_struct_loss(self, h, pc_embeddings, structures, span_mask,
                            masking_rate=0.5):
        """
        Enhanced structure loss with random negative span masking to improve recall
        by addressing the unlabeled entity problem.

        Args:
            h: Span representations
            pc_embeddings: Schema embeddings
            structures: Ground truth structure labels
            span_mask: Mask for valid spans
            masking_rate: Probability of masking negative spans (0.0 to 1.0)
        """
        gold_count_val = min(structures[0], 19)
        struct_proj = self.count_embed(pc_embeddings[1:], gold_count_val)
        scores = torch.einsum('lkd,bpd->bplk', h, struct_proj)

        # Create label tensor matching the shape of scores
        labs = torch.zeros_like(scores)
        for i in range(gold_count_val):
            gold_spans = structures[1][i]
            for k, span in enumerate(gold_spans):
                if span is None or span == (-1, -1):
                    continue
                if isinstance(span, tuple):
                    start, end = span
                    width = end - start
                    if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                        labs[i, k, start, width] = 1
                elif isinstance(span, list):
                    for sub_span in span:
                        if sub_span is None or sub_span == (-1, -1):
                            continue
                        start, end = sub_span
                        width = end - start
                        if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                            labs[i, k, start, width] = 1

        # Apply random masking to negative spans only
        if masking_rate > 0.0:
            # Identify negative spans (where labels are 0)
            negative_spans = (labs == 0)

            # Generate random mask for negative spans
            random_mask = torch.rand_like(scores) < masking_rate

            # Apply masking only to negative spans
            spans_to_mask = negative_spans & random_mask

            # Create loss computation mask (1 = compute loss, 0 = ignore)
            loss_computation_mask = (~spans_to_mask).float()
        else:
            loss_computation_mask = torch.ones_like(scores)

        # Compute loss with masking applied
        loss_struct = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
        loss_struct = loss_struct * loss_computation_mask
        loss_struct = loss_struct.view(loss_struct.shape[0], loss_struct.shape[1], -1) * (~span_mask[0]).float()
        loss_struct = loss_struct.sum()

        return loss_struct

    def forward(
            self,
            records: List[Dict[str, Any]],
            return_individual_losses: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process multiple records in batch for efficient training.

        This method efficiently processes multiple records by:
        1. Batch encoding through the transformer
        2. Computing losses for each sample
        3. Aggregating results

        Parameters
        ----------
        records : List[Dict[str, Any]]
            List of records, each with 'text' and 'schema' keys
        return_individual_losses : bool, default=False
            If True, return individual losses per sample in addition to aggregated losses

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'total_loss': Averaged total loss across batch
            - 'classification_loss': Averaged classification loss
            - 'structure_loss': Averaged structure loss
            - 'count_loss': Averaged count loss
            - 'batch_size': Number of records processed
            - 'individual_losses': (optional) List of loss dicts per sample
        """
        if not records:
            device = next(self.parameters()).device
            result = {
                'total_loss': torch.tensor(0.0, device=device),
                'classification_loss': torch.tensor(0.0, device=device),
                'structure_loss': torch.tensor(0.0, device=device),
                'count_loss': torch.tensor(0.0, device=device),
                'batch_size': 0
            }
            if return_individual_losses:
                result['individual_losses'] = []
            return result

        # Batch encode through processor
        packing_cfg = self._build_packing_config()
        batch_results = self.processor.batch_process_records(
            self.encoder,
            records,
            packing_config=packing_cfg,
            encoder_wrapper=self._apply_pair_attention_encoder,
        )

        # Prepare for loss computation
        all_classification_losses = []
        all_structure_losses = []
        all_count_losses = []
        all_total_losses = []
        individual_loss_records = []

        device = next(self.parameters()).device

        # Process each sample's losses
        valid_samples = 0
        for i in range(len(records)):
            try:
                # Skip if no valid embeddings
                if i >= len(batch_results.token_embeddings) or len(batch_results.token_embeddings[i]) == 0:
                    print(f"Warning: Skipping sample {i} due to missing embeddings")
                    continue

                # Compute span representations for this sample
                # if all element of batch_results.task_types[i] is "classifications" do
                if all(task == "classifications" for task in batch_results.task_types[i]):
                    span_rep = {}
                else:
                    span_rep = self.compute_span_rep(batch_results.token_embeddings[i])

                # Classification loss
                cls_loss = self.classification_loss(
                    batch_results.embs_per_schema[i],
                    batch_results.task_types[i],
                    batch_results.structure_labels[i]
                )
                all_classification_losses.append(cls_loss)

                # Structure and count losses
                struct_loss_total = torch.tensor(0.0, device=device)
                count_loss_total = torch.tensor(0.0, device=device)

                all_counts = []
                all_p_embs = []

                for schema_idx in range(len(batch_results.embs_per_schema[i])):
                    if batch_results.task_types[i][schema_idx] != "classifications":
                        # Handle empty embeddings
                        if not batch_results.embs_per_schema[i][schema_idx]:
                            continue

                        schema_emb = torch.stack(batch_results.embs_per_schema[i][schema_idx])
                        structure = batch_results.structure_labels[i][schema_idx]

                        if structure[0] == 0:
                            loss_struct = torch.tensor(0.0, device=device)
                        else:
                            # Use dynamic masking rate during training
                            loss_struct = self.compute_struct_loss(
                                span_rep["span_rep"],
                                schema_emb,
                                structure,
                                span_rep["span_mask"],
                            )
                        struct_loss_total += loss_struct

                        if batch_results.task_types[i][schema_idx] != "entities":
                            all_counts.append(min(structure[0], 19))
                            all_p_embs.append(schema_emb[0])

                all_structure_losses.append(struct_loss_total)

                # Count loss
                if all_counts and all_p_embs:
                    all_counts = torch.LongTensor(all_counts).to(device)
                    all_p_embs = torch.stack(all_p_embs, dim=0)
                    count_loss = F.cross_entropy(self.count_pred(all_p_embs), all_counts, reduction="sum")
                    count_loss_total += count_loss

                all_count_losses.append(count_loss_total)

                # Total loss for this sample
                sample_total_loss = cls_loss + struct_loss_total + count_loss_total
                all_total_losses.append(sample_total_loss)

                # Store individual losses if requested
                if return_individual_losses:
                    individual_loss_records.append({
                        'total_loss': sample_total_loss.item(),
                        'classification_loss': cls_loss.item(),
                        'structure_loss': struct_loss_total.item(),
                        'count_loss': count_loss_total.item()
                    })

                valid_samples += 1

            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                # Add zero losses for failed samples
                zero_loss = torch.tensor(0.0, device=device)
                all_classification_losses.append(zero_loss)
                all_structure_losses.append(zero_loss)
                all_count_losses.append(zero_loss)
                all_total_losses.append(zero_loss)

                if return_individual_losses:
                    individual_loss_records.append({
                        'total_loss': 0.0,
                        'classification_loss': 0.0,
                        'structure_loss': 0.0,
                        'count_loss': 0.0,
                        'error': str(e)
                    })

        # Handle case where all samples failed
        if valid_samples == 0:
            result = {
                'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'classification_loss': torch.tensor(0.0, device=device),
                'structure_loss': torch.tensor(0.0, device=device),
                'count_loss': torch.tensor(0.0, device=device),
                'batch_size': len(records)
            }
            if return_individual_losses:
                result['individual_losses'] = individual_loss_records
            return result

        # Aggregate losses
        total_classification_loss = torch.stack(all_classification_losses).sum()
        total_structure_loss = torch.stack(all_structure_losses).sum()
        total_count_loss = torch.stack(all_count_losses).sum()
        total_loss = torch.stack(all_total_losses).sum()

        # Average by number of valid samples
        result = {
            'total_loss': total_loss,
            'classification_loss': total_classification_loss,
            'structure_loss': total_structure_loss,
            'count_loss': total_count_loss,
            'batch_size': valid_samples
        }

        # Ensure loss requires gradients for training
        if self.training:
            result['total_loss'].requires_grad_(True)

        if return_individual_losses:
            result['individual_losses'] = individual_loss_records

        return result

    # ---------------------------
    # Hugging Face Compatibility Methods
    # ---------------------------
    def push_to_hub(self, repo_id: str, private: bool = True):
        """
        Push the model (configuration, weights, and tokenizer) to the Hugging Face Hub.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            super().push_to_hub(repo_id=repo_id, save_dir=tmp_dir, private=private)
            # Note: the tokenizer used is from the processor.
            self.processor.tokenizer.push_to_hub(repo_id)

    @classmethod
    def from_pretrained(cls, repo_or_dir, **kwargs):
        """
        Load the model from a directory or repository on the Hugging Face Hub.
        """
        from huggingface_hub import hf_hub_download  # ensure this is available

        def download_or_local(repo_or_dir, filename):
            if os.path.isdir(repo_or_dir):
                return os.path.join(repo_or_dir, filename)
            else:
                return hf_hub_download(repo_or_dir, filename)

        # Load the model configuration.
        config_path = download_or_local(repo_or_dir, "config.json")
        config = cls.config_class.from_pretrained(config_path)

        # Load the encoder configuration.
        encoder_config_path = download_or_local(repo_or_dir, "encoder_config/config.json")
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        # Load the tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)

        # Initialize the model.
        model = cls(config, encoder_config=encoder_config, tokenizer=tokenizer)

        # Load model weights.
        try:
            model_path = download_or_local(repo_or_dir, "model.safetensors")
            model_state_dict = load_file(model_path)
        except Exception:
            model_path = download_or_local(repo_or_dir, "pytorch_model.bin")
            model_state_dict = torch.load(model_path, map_location="cpu")

        # if shape is not equal, fill additional tokens embedding to the state dict given the model
        try:
            if model_state_dict["encoder.embeddings.word_embeddings.weight"].shape[0] != \
                    model.encoder.embeddings.word_embeddings.weight.shape[0]:
                model_state_dict["encoder.embeddings.word_embeddings.weight"] = torch.cat(
                    [
                        model_state_dict["encoder.embeddings.word_embeddings.weight"],
                        torch.randn(model.encoder.embeddings.word_embeddings.weight.shape[0] -
                                    model_state_dict["encoder.embeddings.word_embeddings.weight"].shape[0],
                                    model_state_dict["encoder.embeddings.word_embeddings.weight"].shape[1]) * 0.02
                    ],
                    dim=0
                )
        except KeyError:
            pass

        model.load_state_dict(model_state_dict)

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model configuration, encoder configuration, state dict (using safetensors),
        and tokenizer to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save the main config.
        self.config.save_pretrained(save_directory)

        # Save encoder configuration.
        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        # Save model weights using safetensors.
        model_save_path = os.path.join(save_directory, "model.safetensors")
        save_file(self.state_dict(), model_save_path)

        # Save the tokenizer from the processor.
        self.processor.tokenizer.save_pretrained(save_directory)
