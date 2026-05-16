"""BaseAgent ABC — all agents inherit from this."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base for all LLM agents.

    Agents are pure business logic: they receive data, call the LLM client,
    and return data. They must NOT import FastAPI, Celery, SQLAlchemy,
    or perform any direct file I/O.
    """

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary action."""
        ...
