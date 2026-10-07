# Docker policy server

An optional Dockerfile packages the policy server, the root Python 3.12 environment, and its Transformers patch. Install [Docker with the Buildx and Compose plugins](https://docs.docker.com/engine/install/ubuntu/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on a Linux host with a CUDA 12.8-compatible NVIDIA driver. The Dockerfile requires [BuildKit](https://docs.docker.com/build/buildkit/) for its dependency cache.

From the repository root:

```bash
docker buildx build --load -t policy-server -f scripts/docker/serve_policy.Dockerfile .
mkdir -p "$HOME/.cache/robot_policy"
docker run --rm --gpus all --network host \
  --mount type=bind,src=/absolute/path/to/checkpoint/step,dst=/checkpoint,readonly \
  --mount type=bind,src="$HOME/.cache/robot_policy",dst=/model_assets \
  -e DATA_HOME=/model_assets -e HF_HOME=/model_assets/huggingface \
  -e SERVER_ARGS="--env LIBERO --port 8001 policy:checkpoint --policy.config=stage3_finetuning_libero --policy.dir=/checkpoint" \
  policy-server
```

Replace the checkpoint path with an existing absolute path to a saved step directory containing the policy weights and their original normalization assets. Set the policy configuration to match that checkpoint. `SERVER_ARGS` supplies the policy server's CLI arguments; the example listens on port 8001.

Alternatively, use Compose from the repository root:

```bash
POLICY_DIR=/absolute/path/to/checkpoint/step \
  docker compose -f scripts/docker/compose.yml up --build
```

Compose defaults to `CONFIG=stage3_finetuning_libero` and `POLICY_PORT=8001`. Set those environment variables for another configuration or port. `DATA_HOME` selects the host cache directory and defaults to `~/.cache/robot_policy`; Hugging Face downloads are also cached there. The checkpoint is mounted read-only at `/checkpoint`.

Both commands use the code built into the image. Rebuild the image after changing the source. The build excludes external benchmark checkouts, datasets, checkpoints, caches, and local environments. Run the LIBERO simulator separately using [its dedicated environment](../examples/libero/README.md), and wait for the server's listening message before starting it.
