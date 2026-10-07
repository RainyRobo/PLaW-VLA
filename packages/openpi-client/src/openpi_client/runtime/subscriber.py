# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
import abc


class Subscriber(abc.ABC):
    """Subscribes to events in the runtime.

    Subscribers can be used to save data, visualize, etc.
    """

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Called when an episode starts."""

    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        """Append a step to the episode."""

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Called when an episode ends."""
