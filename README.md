<h1 align="center">PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.12285"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.12285-B31B1B?style=flat&logo=arxiv&logoColor=white"></a>
  <a href="https://rainyrobo.github.io/PLaW-VLA/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2563EB?style=flat"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-16A34A?style=flat"></a>
</p>

<p align="center">
  <strong>Yu Liu</strong><sup>1,2,*</sup>&nbsp;&nbsp;
  <strong>Hetian Guo</strong><sup>1,*</sup>&nbsp;&nbsp;
  Tianlv Huang<sup>1</sup>&nbsp;&nbsp;
  Ziyi Cai<sup>3</sup>&nbsp;&nbsp;
  Wudi Chen<sup>1</sup>&nbsp;&nbsp;
  Hantang Wang<sup>4</sup>&nbsp;&nbsp;
  Qiutong Liu<sup>4</sup>
  <br>
  Yingzhi Peng<sup>5</sup>&nbsp;&nbsp;
  Wei Han<sup>1</sup>&nbsp;&nbsp;
  Peijun Tang<sup>2,†</sup>&nbsp;&nbsp;
  Jianan Wang<sup>2,†</sup>&nbsp;&nbsp;
  Zipei Fan<sup>1,‡</sup>&nbsp;&nbsp;
  Zhiyuan Zha<sup>1</sup>&nbsp;&nbsp;
  Xuan Song<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>&nbsp;Jilin University&nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup>&nbsp;Astribot&nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>3</sup>&nbsp;Harbin Institute of Technology, Shenzhen
  <br>
  <sup>4</sup>&nbsp;The Hong Kong Polytechnic University&nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>5</sup>&nbsp;The University of Tokyo
</p>

<p align="center">
  <sup>*</sup>Equal contribution&nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>†</sup>Project leads&nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>‡</sup>Corresponding author
</p>

<p align="center">
  <img src="assets/teaser.png" alt="Overview of PLaW-VLA and its evaluation results" width="100%">
</p>

PLaW-VLA is a vision-language-action framework that learns to predict future
visual states in the latent space of a frozen V-JEPA 2 encoder. Its
Mixture-of-Transformers architecture integrates vision-language, latent
world-model, and action experts, allowing continuous action generation to use
both the current observation and predicted future representations.

## Installation



### Requirements

- Linux x86_64
- NVIDIA GPU with a CUDA 12.8-compatible driver
- Python 3.12
- Git, FFmpeg, EGL, and OpenGL runtime libraries

On Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg libgl1 libegl1 libglib2.0-0
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
create the training environment:

```bash
git clone https://github.com/RainyRobo/PLaW-VLA.git
cd PLaW-VLA
git submodule update --init third_party/libero
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.12 --frozen
bash scripts/setup/install_transformers_patch.sh
```

If the system Python installation does not include development headers, use
uv's managed Python:

```bash
uv python install 3.12
UV_PYTHON_PREFERENCE=only-managed \
GIT_LFS_SKIP_SMUDGE=1 \
uv sync --python 3.12 --frozen
bash scripts/setup/install_transformers_patch.sh
```

Training uses PyTorch Distributed Data Parallel. `NUM_GPUS` selects the visible
GPU count used by a training wrapper, and the global batch size must be
divisible by that value. Dataset conversion and simulator clients use their
own environments as documented under [`scripts/data`](scripts/data) and
[`examples`](examples).

## Models and data


| Resource                                | Source                                                                                  | Availability |
| --------------------------------------- | --------------------------------------------------------------------------------------- | ------------ |
| π₀.₅ foundation checkpoint              | `gs://openpi-assets/checkpoints/pi05_base`                                              | Available    |
| V-JEPA 2 encoder                        | [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | Available    |
| LIBERO EEF dataset                      | [RainyBot/libero_v3_eef](https://huggingface.co/datasets/RainyBot/libero_v3_eef)        | Available    |
| Official PLaW-VLA pretrained checkpoint | Release planned                                                                         | Pending      |


The repository provides the conversion and validation pipelines for the
supported source datasets. Refer to the [data guide](docs/data.md) for source
layouts and converter commands. The exact paper mixture additionally depends
on the author-verified source manifest described in that guide.

## Training workflows

PLaW-VLA supports two official workflows:

1. **Downstream task fine-tuning** adapts a pretrained policy to a target
  dataset and embodiment. LIBERO is provided as the complete reference
   example.
2. **Full training** reproduces the Stage I → Stage II → Stage III pipeline.



### 1. Downstream task fine-tuning

The downstream workflow starts from a compatible foundation or PLaW-VLA
checkpoint and applies the Stage III recipe to task-specific demonstrations.
The following command uses LIBERO as the reference implementation:

```bash
STAGE3_BASE_SOURCE=auto \
NUM_GPUS=8 EXP_NAME=libero_finetune \
bash scripts/train/run_stage3_libero_quickstart.sh
```

> **Current release:** the official PLaW-VLA pretrained checkpoint is not yet
> available. `auto` therefore loads the public π₀.₅ foundation weights and
> initializes the PLaW-VLA world-model modules from the Stage III
> configuration.
>
> **After the pretrained checkpoint release:** the same command will
> automatically load the complete official PLaW-VLA checkpoint, including the
> pretrained world model. No command change will be required.

For the LIBERO example, the entrypoint prepares the initialization checkpoint,
V-JEPA 2, tokenizer, EEF dataset, and matching normalization statistics. The
official Stage III recipe uses 50,000 optimization steps, a global batch size
of 256, a 10,000-step warmup, and checkpoints every 5,000 steps. Outputs are
written to:

```text
checkpoints/stage3_finetuning_libero/libero_finetune/<step>/
```

Resume the same experiment with `--resume` while preserving the original
source, process count, batch size, and training options.

Adapting another simulator or robot follows the same workflow: define its
dataset configuration and normalization contract, encode observations into
the canonical policy state, and implement an execution adapter from canonical
absolute EEF targets to the native controller. These embodiment-specific
boundaries do not require changes to the model architecture.

### 2. Full Stage I–III training

Prepare the five source datasets and Stage II normalization statistics by
following the [end-to-end pretraining guide](docs/pretraining.md). Run the
stages in order:

#### Stage I: World-model pretraining

```bash
export DATA_ROOT="$PWD/data/pretrain"
NUM_GPUS=8 EXP_NAME=stage1_world_model \
bash scripts/train/run_stage1_world_model_pretraining.sh
```



#### Stage II: Joint world-model and action pretraining

```bash
export DATA_ROOT="$PWD/data/pretrain"
NUM_GPUS=8 STAGE1_EXP_NAME=stage1_world_model EXP_NAME=stage2_joint \
bash scripts/train/run_stage2_pretraining.sh
```



#### Stage III: Task fine-tuning

```bash
NUM_GPUS=8 STAGE2_EXP_NAME=stage2_joint EXP_NAME=libero \
bash scripts/train/run_stage3_finetuning_libero.sh
```


| Stage     | Objective                                   | Steps   | Global batch |
| --------- | ------------------------------------------- | ------- | ------------ |
| Stage I   | Latent world-model alignment                | 100,000 | 512          |
| Stage II  | Joint future prediction and action learning | 100,000 | 256          |
| Stage III | LIBERO task adaptation                      | 50,000  | 256          |


Stage II selects the latest valid checkpoint from `STAGE1_EXP_NAME`; Stage III
does the same for `STAGE2_EXP_NAME`. Set `STAGE2_INIT_WEIGHT` or
`STAGE3_INIT_WEIGHT` to select a specific checkpoint step. Keep each complete
step directory because resume and inference require its saved weights,
training state, and normalization assets.

## LIBERO evaluation

Install the dedicated LIBERO client environment:

```bash
uv sync --project examples/libero --python 3.8 --frozen
```

Start the policy server from a Stage III checkpoint:

```bash
.venv/bin/python scripts/serve/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config=stage3_finetuning_libero \
  --policy.dir=/path/to/libero/checkpoint/step
```

Run evaluation in a second terminal:

```bash
MUJOCO_GL=egl \
uv run --project examples/libero --frozen \
python examples/libero/main.py \
  --task-suite-name libero_spatial \
  --seed 42
```

Supported suites are `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. The server returns timestamped absolute EEF targets; the client
interpolates them at the simulator control rate and converts each live pose
error into a robosuite OSC command.

For full-suite scheduling, task selection, video recording, and result
aggregation, see the [LIBERO evaluation guide](examples/libero/README.md).
The protocol and execution contract are documented in
[remote inference](docs/remote_inference.md).

## Documentation


| Guide                                        | Contents                                                             |
| -------------------------------------------- | -------------------------------------------------------------------- |
| [Data preparation](docs/data.md)             | Source layouts, acquisition requirements, conversion, and validation |
| [Pretraining](docs/pretraining.md)           | End-to-end Stage I–III data and training workflow                    |
| [Normalization](docs/norm_stats.md)          | State/action conventions, statistics, and checkpoint assets          |
| [Remote inference](docs/remote_inference.md) | Policy server contract and robot execution adapters                  |
| [Optional benchmarks](docs/benchmarks.md)    | RoboTwin and LIBERO-Plus setup and evaluation                        |
| [Contributing](CONTRIBUTING.md)              | Development workflow and contribution requirements                   |
| [Security](SECURITY.md)                      | Vulnerability reporting policy                                       |




## Repository structure

```text
PLaW-VLA/
├── src/plawvla/              # Models, policies, training, serving, and datasets
├── packages/plawvla-client/  # Lightweight WebSocket and execution client
├── scripts/
│   ├── data/                 # Dataset conversion pipelines
│   ├── setup/                # Asset preparation and checkpoint conversion
│   ├── train/                # Stage I–III training entrypoints
│   └── serve/                # Policy servers
├── examples/                 # LIBERO, LIBERO-Plus, and RoboTwin clients
├── docs/                     # Data, training, normalization, and inference guides
└── third_party/              # Pinned external benchmark repositories
```



## Release status


| Component                                     | Status          |
| --------------------------------------------- | --------------- |
| Model and distributed training implementation | Available       |
| Dataset converters and validation tools       | Available       |
| Stage I–III checkpoint handoff                | Available       |
| LIBERO training and evaluation                | Available       |
| Official PLaW-VLA pretrained checkpoint       | Pending release |




## Citation

If you use PLaW-VLA in your work, please cite:

```bibtex
@inproceedings{liu2026plawvla,
  title     = {PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies},
  author    = {Liu, Yu and Guo, Hetian and Huang, Tianlv and Cai, Ziyi and Chen, Wudi and Wang, Hantang and Liu, Qiutong and Peng, Yingzhi and Han, Wei and Tang, Peijun and Wang, Jianan and Fan, Zipei and Zha, Zhiyuan and Song, Xuan},
  booktitle = {Conference on Robot Learning},
  year      = {2026}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

The PLaW-VLA source code is released under the
[Apache License 2.0](LICENSE). Third-party notices and exceptions are listed
in [`NOTICE`](NOTICE) and [`LICENSES`](LICENSES). Model weights, datasets, and
external benchmark assets may be subject to separate terms.

## Acknowledgments

This project builds on [openpi](https://github.com/Physical-Intelligence/openpi),
[V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[LeRobot](https://github.com/huggingface/lerobot), and
[any4lerobot](https://github.com/Tavish9/any4lerobot). We thank their authors
for releasing the corresponding models, environments, and tooling.
