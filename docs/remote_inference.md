# Remote inference

The policy server runs in the root environment on a GPU machine. Simulator and robot clients use the lightweight `openpi_client` package and the same WebSocket protocol.

## Server

```bash
.venv/bin/python scripts/serve_policy.py --env LIBERO --port 8001 policy:checkpoint \
  --policy.config=stage3_finetuning_libero --policy.dir=/path/to/checkpoint/step
```

Use a checkpoint with its original normalization assets and matching configuration. Wait for the server to report that it is listening before starting a client. The server publishes an `input_spec` containing camera names, temporal offsets, state and prompt keys, and action/gripper conventions.

## Client

Install the client package into the simulator or robot environment:

```bash
pip install -e /path/to/PLaW-VLA/packages/openpi-client
```

```python
from openpi_client import websocket_client_policy

with websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8001) as client:
    metadata = client.get_server_metadata()
    spec = metadata["input_spec"]

    # Construct these values from the environment using the published spec.
    # LIBERO front_history is uint8 [T, H, W, 3] at history_step_offsets;
    # wrist_rgb is uint8 [H, W, 3]. state contains xyz, rotation-vector,
    # and two gripper finger positions, as described by state_gripper_format.
    observation = {
        "observation/image": front_history,
        "observation/wrist_image": wrist_rgb,
        "observation/state": state,
        "prompt": instruction,
    }
    actions = client.infer(observation)["actions"]
```

The client waits up to 30 seconds to connect and 60 seconds for an inference response. Set `connect_timeout` and `inference_timeout` when constructing it to adjust these limits. The LIBERO and LIBERO-Plus clients expose the corresponding `--connect-timeout` and `--inference-timeout` options. The context manager closes the connection when the block exits.

Images should match the training orientation and can be resized with `openpi_client.image_tools.resize_with_pad`. Normalization happens on the server. Construct temporal history at the published offsets rather than repeating a guessed number of frames. The [LIBERO client](../examples/libero/main.py) provides a complete implementation, including episode boundaries and action conversion.

The response is an action chunk. Execute only the selected replanning window before sending the next observation. Interpret actions using the benchmark adapter and the published gripper contract; they are not a universal robot control format.
