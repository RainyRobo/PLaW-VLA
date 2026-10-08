# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
import inspect
import logging
import math
import time
import urllib.parse
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from plawvla_client import base_policy as _base_policy
from plawvla_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_timeout: float = 30.0,
        inference_timeout: float = 60.0,
    ) -> None:
        """Connect to a server with bounded startup and response waits, in seconds."""
        for name, value in (("connect_timeout", connect_timeout), ("inference_timeout", inference_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not host.startswith(("ws://", "wss://")):
            if host.count(":") > 1 and not host.startswith("["):
                host = f"[{host}]"
            host = f"ws://{host}"
        parsed = urllib.parse.urlsplit(host)
        if port is not None:
            if not 0 < port < 65536:
                raise ValueError("port must be between 1 and 65535")
            if parsed.port is not None and parsed.port != port:
                raise ValueError("host URL and port specify different ports")
            if parsed.port is None:
                parsed = parsed._replace(netloc=f"{parsed.netloc}:{port}")
        self._uri = urllib.parse.urlunsplit(parsed)
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._connect_timeout = connect_timeout
        self._inference_timeout = inference_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + self._connect_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Could not connect to {self._uri} within {self._connect_timeout:g} seconds")
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                # Older sync clients do not send keepalive pings and do not accept
                # this argument. Disable them when the installed version supports it.
                keepalive = (
                    {"ping_interval": None}
                    if "ping_interval" in inspect.signature(websockets.sync.client.connect).parameters
                    else {}
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=remaining,
                    close_timeout=1.0,
                    **keepalive,
                )
                try:
                    metadata = conn.recv(timeout=max(0, deadline - time.monotonic()))
                    if isinstance(metadata, str):
                        raise RuntimeError(f"Error in inference server:\n{metadata}")
                    metadata = msgpack_numpy.unpackb(metadata)
                    if not isinstance(metadata, dict):
                        raise ValueError("Server metadata must be a dictionary")
                except BaseException:
                    conn.close()
                    raise
                return conn, metadata
            except ConnectionRefusedError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Could not connect to {self._uri} within {self._connect_timeout:g} seconds"
                    ) from exc
                time.sleep(min(0.25, remaining))

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._ws is None:
            raise RuntimeError("Policy connection is closed")
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv(timeout=self._inference_timeout)
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            response = msgpack_numpy.unpackb(response)
            if not isinstance(response, dict):
                raise ValueError("Server inference response must be a dictionary")
            return response
        except TimeoutError as exc:
            # A late response cannot be reused for a later observation.
            self.close()
            raise TimeoutError(
                f"Server {self._uri} did not respond within {self._inference_timeout:g} seconds"
            ) from exc
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the connection. Repeated calls are harmless."""
        conn, self._ws = self._ws, None
        if conn is not None:
            conn.close()

    def __enter__(self) -> "WebsocketClientPolicy":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @override
    def reset(self) -> None:
        pass
