# Simple policy client

This example measures inference timing using synthetic LIBERO observations and the server's temporal sampling metadata. It does not execute a simulator task.

Start a LIBERO policy server as described in [the root README](../../README.md#libero-evaluation), then run from the repository root:

```bash
.venv/bin/python examples/simple_client/main.py --host localhost --port 8001 --num-steps 20
```

Use `--timing-file /path/to/timing.parquet` to save timing samples. For real evaluation, use [the LIBERO client](../libero/README.md). To embed the WebSocket client in another Python environment, follow [remote inference](../../docs/remote_inference.md).
