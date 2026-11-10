# Training sessions and managed runs

`TrainingSession` exposes bounded operations over the same native trainer engine:
`forward`, `forward_backward`, `optim_step`, `save_state`, `load_state`, and
`export_weights`. It supports span and boundary models, full-weight training,
and one trainable default PEFT LoRA adapter. A native trained-model export can
start another run. A preloaded adapter must match its declared rank, alpha,
dropout, targets and DoRA setting; it is continued without reinitialization.

```python
from gliner2 import AutoExtractor
from gliner2.training import ManagedTraining, ManagedTrainingPlan, TrainingConfig, TrainingSession

model = AutoExtractor.from_pretrained("/approved/local-model")
config = TrainingConfig(
    output_dir="/run", num_workers=0, fp16=False, bf16=False,
    batch_size=2, gradient_accumulation_steps=2, num_epochs=3,
    eval_strategy="epoch", early_stopping=True, early_stopping_patience=2,
)
plan = ManagedTrainingPlan(examples, config, architecture=model.architecture)
session = TrainingSession(model, config, max_steps=plan.max_steps)
run = ManagedTraining(session, plan, eval_examples=validation)
try:
    result = run.run(on_step=lambda step, metrics: print(step, metrics))
    run.save_state("/new-immutable-checkpoint")
    run.export_weights("/selected-native-model")
finally:
    run.close()
```

The plan is also the contract for a remote dispatcher: submit `plan.batch(i)`
to `forward_backward`, then submit `optim_step` iff `plan.is_update(i)`. Persist
the next batch index only after the corresponding operations commit. `batch`,
`is_update`, `is_epoch_end`, `epoch`, `batch_count`, `max_steps`, and `fingerprint`
are public. The plan includes final partial batches rather than dropping data;
partial accumulation is normalized and flushed at each epoch boundary. The
native trainer owns loss, backward, parameter grouping, gradient clipping,
optimizer, scheduler, preprocessing and architecture-specific loss schedules.
There is no second optimizer implementation.

The managed driver supports epoch/step validation, best-validation checkpoint
selection, early stopping and progress callbacks. Its result uses the native
trainer's summary/history keys. Best export loads a detached CPU copy, leaving
live parameters and optimizer state untouched. Callers own artifact publication,
health reporting, authorization and distributed job/session recovery. Direct
batch operations return the native scalar batch loss, not fabricated per-row
losses.

A session checkpoint contains all model weights, optimizer moments, scheduler,
scaler, pending gradients and Python/NumPy/Torch/CUDA RNG state. Managed
checkpoints add a fingerprinted dataset plan, exact cursor, evaluation/best
state, history and early-stopping counters. Restore requires a fresh session,
the same training/model/sampling configuration and device type. A changed
microbatch plan is rejected; an OOM must not silently resume with a different
optimizer trajectory. Checkpoints are trusted service-owned files loaded with
`weights_only=True`, never arbitrary client pickle uploads.

One process owns one session; all operations must be serialized because native
preprocessing and Torch RNGs are process-global. `num_workers=0` is required.
Distributed training, graph compilation, lossy sample/capacity skipping and
cross-device continuation are rejected. CUDA mixed precision is supported by
the engine; CPU mixed-precision requests fail instead of silently downgrading.
Explicit precision avoids architecture-dependent defaults. Boundary fp16 is not
part of the supported GPU qualification matrix.

## Qualification

CPU tests use real offline tiny encoder/head/tokenizer fixtures. They verify
span/boundary × full/LoRA managed/direct equality, exact checkpoint continuation
including incomplete accumulation, native inference export, trained-model and
preloaded-adapter continuation, evaluation/best selection and early stopping.
They do not measure production throughput, model quality or cloud provisioning.

```sh
pytest -q tests/test_training_session.py tests/test_managed_training_session.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -q tests/test_training_session_cuda.py
```

The GPU suite runs real CUDA span fp32/fp16/bf16 and boundary fp32/bf16 for both
full weights and LoRA. A CPU skip is not GPU qualification. A T4 should run six
cases and skip four bf16 cases because the device lacks native bf16 support.

`ManagedTrainingPlan.to_manifest(eval_examples=...)` exports a content-addressed
JSON schedule with exact row indices, update boundaries, validation boundaries,
and selection policy. A remote controller can replay this schedule through its
existing durable operation queue, assigning stable operation keys from the run
identity and manifest fingerprint. The manifest contains training examples;
store it with the same authorization and retention as the source dataset.

`TrainingSession.load_weights(checkpoint)` starts a new training run from a
complete checkpoint's model weights while keeping the new run's fresh optimizer,
schedule, random state, and zero pending gradients. `load_state(checkpoint)` instead
restores all continuation state and requires the same training configuration.
An optional `trainer_factory` can supply a subclass of the native trainer; its
`session_state_dict` / `load_session_state_dict` hooks include recovery buffers in
the same complete checkpoint. The session never adds a second optimizer loop.

Planning requires no Torch, NumPy, Transformers, or model weights:

```python
from gliner2.training import InputExample, TrainingConfig, ManagedTrainingPlan

config = TrainingConfig(num_workers=0, fp16=False, bf16=False)
plan = ManagedTrainingPlan(examples, config, architecture="boundary")
manifest = plan.to_manifest(eval_examples=validation_examples)
```

The new managed planner owns a deterministic Python random stream for shuffling
and length grouping. It does not reuse the legacy trainer's Torch sampler stream;
the legacy sampler remains unchanged. Hardware precision checks happen when the
training engine is created, so a CPU control plane can plan a CUDA run.
