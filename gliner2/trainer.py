import json
import random
from typing import Union, List

import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import torch

# Import apex if available (for fp16 support)
if transformers.utils.is_apex_available():
    from apex import amp

# Import sagemaker if available
from transformers.trainer import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward


class ExtractorDataset(Dataset):
    """
    A Dataset for loading JSONL data tailored for the Extractor model.
    Accepts a single file path or a list of file paths.
    """

    def __init__(self, data_paths: Union[str, List[str]]):
        if isinstance(data_paths, str):
            data_paths = [data_paths]  # Ensure it's a list

        # log number of files
        print(f"Loading {len(data_paths)} files for training.")

        self.data = []
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as f:
                self.data.extend([json.loads(line) for line in f])

        # shuffle the data
        random.shuffle(self.data)

        # log number of records
        print(f"Loaded {len(self.data)} records from {len(data_paths)} files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data[idx]
        # Map keys to what your model expects.
        return record["input"], record["output"]


class ExtractorDataCollator:
    """
    Data collator for the Extractor model.
    """

    def __call__(self, batch):
        # return batch  # NOTE: main version with evaluation error: "AttributeError: 'list' object has no attribute 'get'"

        # batch is a list of tuples (input, output) from ExtractorDataset
        # We want to convert it to a list of dicts for the model
        records = [{"text": x[0], "schema": x[1]} for x in batch]
        return {"records": records}


class ExtractorTrainer(Trainer):
    """
    A Trainer with customized optimizer and training step tailored for the Extractor model.
    Now supports an option to freeze everything except `model.classifier`.
    Includes FP16 support.
    """

    def __init__(
            self,
            encoder_lr,
            custom_lr,
            weight_decay,
            finetune_classifier: bool = False,
            **kwargs
    ):
        """
        Args:
            encoder_lr (float): learning rate for encoder parameters (ignored if finetune_classifier=True)
            custom_lr (float): learning rate for non-encoder parameters (e.g., classifier)
            weight_decay (float): weight decay for all parameter groups
            finetune_classifier (bool): if True, freeze all parameters except `model.classifier`
        """
        self.encoder_lr = encoder_lr
        self.custom_lr = custom_lr
        self.custom_weight_decay = weight_decay
        self.finetune_classifier = finetune_classifier

        super().__init__(**kwargs)

        if self.finetune_classifier:
            # Freeze all parameters except classifier
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def create_optimizer(self):
        """
        Create an optimizer with separate parameter groups.
        If finetune_classifier=True, only classifier params go into optimizer.
        Otherwise, use two groups: encoder and everything else.
        """

        if self.finetune_classifier:
            # log
            print("Finetuning classifier only: freezing all other parameters.")
            # Only include classifier parameters in optimizer
            classifier_params = [
                p for n, p in self.model.named_parameters()
                if n.startswith("classifier") and p.requires_grad
            ]
            if not classifier_params:
                raise ValueError("No trainable parameters found in `model.classifier`.")
            optimizer_grouped_parameters = [
                {
                    "params": classifier_params,
                    "lr": self.custom_lr,
                    "weight_decay": self.custom_weight_decay,
                }
            ]
        else:
            # Full fine-tuning: encoder and others separated
            # encoder parameters
            encoder_params = list(self.model.encoder.parameters())
            # everything else (including classifier, count layers, etc.)
            other_params = [
                param
                for name, param in self.model.named_parameters()
                if "encoder" not in name and param.requires_grad
            ]
            optimizer_grouped_parameters = [
                {"params": encoder_params, "lr": self.encoder_lr, "weight_decay": self.custom_weight_decay},
                {"params": other_params, "lr": self.custom_lr, "weight_decay": self.custom_weight_decay},
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def compute_loss(self, model, inputs, return_outputs: bool = False, *args, **kwargs):
        """
        Compute the objective for a Hugging-Face Trainer batch **once** with
        `model.process_records` instead of looping over `model.process_record`.
        """
        # Ensure training-time behaviour (negative-span masking, dropout, etc.).
        model.train()
        model.processor.change_mode(is_training=True)

        # 1. Pack HF `inputs` -> List[{"text": str, "schema": dict}]
        # batch_records = [{"text": rec[0], "schema": rec[1]} for rec in inputs]  # NOTE: main version with evaluation error: "AttributeError: 'list' object has no attribute 'get'"
        
        # inputs["records"] has the format [{"text": str, "schema": dict}]
        batch_records = inputs["records"]

        # 2. Forward pass in one call
        batch_out = model.forward(
            batch_records,
            return_individual_losses=False
        )

        # 3. Derive the loss the Trainer expects
        if batch_out["batch_size"] == 0:  # every sample failed – rare
            device = next(model.parameters()).device
            # Return a zero loss tensor that requires gradients
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # forward returns a **sum** over samples
            loss = batch_out["total_loss"]
        
        return (loss, batch_out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs with FP16 support and CUDA OOM handling.

        Args:
            model: The model to train.
            inputs: The inputs and targets of the model.
            num_items_in_batch: Number of items in the batch (optional, for compatibility).
        """
        model.train()

        try:
            inputs = self._prepare_inputs(inputs)

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            del inputs
            torch.cuda.empty_cache()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            return loss.detach() / self.args.gradient_accumulation_steps

        except Exception as e:
            print(f"Skipping iteration due to error: {e}")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # Safely get device for DataParallel or normal model
            _model = getattr(model, "module", model)
            device = next(_model.parameters()).device
            return torch.tensor(0.0, requires_grad=True, device=device)
