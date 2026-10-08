# Remote inference

The policy server runs in the root environment on a GPU machine. Simulator and robot clients use the lightweight `plawvla_client` package and the same WebSocket protocol.

## Server

```bash
.venv/bin/python scripts/serve/serve_policy.py --env LIBERO --port 8001 policy:checkpoint \
  --policy.config=stage3_finetuning_libero --policy.dir=/path/to/checkpoint/step
```

Use a checkpoint with its original normalization assets and matching configuration. Wait for the server to report that it is listening before starting a client. The server publishes an `input_spec` containing camera names, temporal offsets, state and prompt keys, and action/gripper conventions.

## Client

Install the client package into the simulator or robot environment:

```bash
pip install -e /path/to/PLaW-VLA/packages/plawvla-client
```

```python
from plawvla_client import websocket_client_policy

with websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8001) as client:
    metadata = client.get_server_metadata()
    spec = metadata["input_spec"]

    # Construct these values from the environment using the published spec.
    # LIBERO front_history is uint8 [T, H, W, 3] sampled at
    # input_spec.history_time_offsets_s relative to the current timestamp;
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

The server uses eager execution by default. To enable PyTorch compilation, prefix its command with `TORCH_COMPILE_MODE=max-autotune` and construct the client with `inference_timeout=600` to allow for compilation during the first inference.

Images should match the training orientation and can be resized with `plawvla_client.image_tools.resize_with_pad`. Normalization happens on the server. Construct temporal history using `history_time_offsets_s` rather than interpreting policy steps as environment steps. LIBERO-family servers return timestamped 8D absolute EEF targets (`xyz`, `wxyz`, open fraction); the lightweight client execution adapter interpolates them at the environment control rate and converts each live pose error into the native controller command. The [LIBERO client](../examples/libero/main.py) provides the complete implementation.

The response is an action target trajectory. Execute only the selected replanning duration before sending the next observation. The target representation is shared; each benchmark or robot supplies its own execution adapter for OSC, IK, joint, velocity, or hardware-specific control.

## Adding a simulator or robot

Do not add controller scaling, coordinate-frame guesses, or hardware timing to
the model or policy server. A new embodiment needs two explicit boundary
components:

1. **Observation encoder:** convert the live robot state into the canonical
   state declared by `input_spec`, including pose frame, `wxyz` quaternion,
   arm order, and open-high gripper fraction.
2. **Execution adapter:** convert each canonical absolute EEF target into the
   native controller input using the **current live state**. Fixed-rate
   controllers may interpolate `action_target_time_offsets_s`; waypoint/IK
   controllers must declare their own timing semantics.

The client must fail closed if pose frame, quaternion order, gripper range or
direction, arm count, wire dimension, or timing is unknown. LIBERO's
`RobosuiteOSCAdapter` and RoboTwin's `RobotwinAbsoluteEefAdapter` are reference
implementations for fixed-rate OSC and waypoint-based IK respectively.
