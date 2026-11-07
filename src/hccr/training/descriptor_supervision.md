# Descriptor supervision

This runner compares a fine-tuned CNN-only control (`none`) with train-only HOG
and image-topology supervision (`hog`, `tda`). `both` is available for a later
ablation. The YAML has a dedicated schema; use the script below rather than the
general experiment command. Importing the modules or reading the config does not
start training.

All variants start from the same source checkpoint, update the complete CNN and
CosFace classifier, use the same shuffled full training split and validation-only
selection, and preserve the source arithmetic. The 20-epoch pilot uses clean
images. The 100-epoch FP32 configuration uses CUDA image caching and GPU affine
augmentation. Its HOG target follows the augmented image. TDA uses a separate
clean view with matching cached targets, shared CNN weights and frozen BatchNorm
statistics for that pass. The default auxiliary weights are provisional.

HOG uses the existing 96x96, cell-8, 9-bin L2-Hys descriptor. A spatial stage-2 head
predicts 11x11 blocks of 36 values, with MSE averaged over samples and dimensions.
For width 80 the train-only head has 5,796 parameters. Its target is detached; its
stage-2 input keeps gradients. Stage 3 and the classifier also receive the main
classification gradient.

TDA uses lower-star cubical H0/H1 of the **preprocessed grayscale image**, not
CNN activation maps. With white=1/black=0, dark ink enters the sublevel filtration
first. It predicts 12 persistence statistics after excluding the two fixed
essential counts. The 320-to-12 head has 3,852 parameters. Mean/std and constant
dimension masking are fitted only on training samples. The loss is mean SmoothL1
over active target dimensions. This is descriptor regression, not differentiable
persistent-homology matching of a student feature map.

Run commands from the repository root, after reviewing the source checkpoint,
output paths and settings in `configs/experiment/descriptor_supervision.yaml`:

```powershell
# These commands are instructions for a future run; they start real work.
.\.venv\Scripts\python.exe scripts/train_descriptor_supervision.py prepare-tda --method tda
.\.venv\Scripts\python.exe scripts/train_descriptor_supervision.py train --method none
.\.venv\Scripts\python.exe scripts/train_descriptor_supervision.py train --method hog
.\.venv\Scripts\python.exe scripts/train_descriptor_supervision.py train --method tda
```

Target preparation computes train and validation image targets without training
the CNN. Training needs only the train target cache; validation selection uses
classification accuracy, breaking ties by classification cross entropy. Cache
receipts pin the manifest, ordered sample IDs, preprocessing, implementation,
dependency versions and target checksum. Completed caches are immutable. If an
extraction was interrupted before its receipt was published, choose a new cache
directory. CNN embeddings are never cached during fine-tuning.

Each epoch consumes all samples, retains the partial final batch and records the
sample and optimizer-step counts. Auxiliary loss weights ramp to their configured
values over the first two epochs. BatchNorm is updated during training, including
the existing partial-batch momentum adjustment. There is no early stopping or
automatic test/ICDAR evaluation.

Resume an identical run with:

```powershell
.\.venv\Scripts\python.exe scripts/train_descriptor_supervision.py train --method hog --resume
```

`latest.pt` stores CNN, auxiliary heads, optimizer, scheduler, selected CNN state,
and Torch CPU/CUDA RNG states. A failed epoch is replayed from its start. Config,
source, data, code or dependency changes reject resume. Do not change the epoch
budget of an existing cosine schedule. A new run requires a new output directory.

At completion, `checkpoint.pt`, `checkpoint_metadata.json`, and `labels.json`
form a normal CNN artifact readable by `load_model_from_run`. The deployed state
has no descriptor extractor or auxiliary head. Epoch zero is an eligible fallback
if fine-tuning does not improve validation. `latest.pt` still retains the actual
trained model for diagnosis. `result.json` records the selected epoch, deployment
and auxiliary parameter counts; `curves.json` records every trained epoch.

Correctness checks use tiny synthetic LMDB datasets, including actual backward
steps, rather than production training:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_descriptor_supervision.py -q
```

The tests verify auxiliary gradients reach the backbone, HOG block ordering, a
known filled-square/ring topology example, train-only scaling, cache tamper
rejection, normal CNN export, restart equivalence and BF16/FP32 arithmetic when
CUDA is available. These checks do not establish recognition accuracy benefits.
