# Docker policy server

An optional Dockerfile packages the root Python 3.12 environment and its Transformers patch. Install Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on a Linux host with a CUDA 12.8-compatible NVIDIA driver.

From the repository root:

```bash
docker build -t policy-server -f scripts/docker/serve_policy.Dockerfile .
docker run --rm --gpus all --network host \
  -v /path/to/checkpoint/step:/checkpoint:ro \
  -e SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config=stage3_finetuning_libero --policy.dir=/checkpoint" \
  policy-server
```

The Docker build excludes external benchmark checkouts, datasets, checkpoints, caches, and local environments. Downloaded inference resources are cached inside the running container; mount a persistent cache if needed. Run the LIBERO simulator separately using [its dedicated environment](../examples/libero/README.md).
