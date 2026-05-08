from __future__ import annotations

from typing import Callable

from market_signal_engine.agents.base import AgentPerformance, BaseAgent

AgentFactory = Callable[[], BaseAgent]


class AgentRegistry:
    """Central registry for all 26 agents.

    Supports lazy instantiation via factory functions so agents are only
    created when first accessed — keeps cold-start fast.
    """

    def __init__(self) -> None:
        self._factories: dict[str, AgentFactory] = {}
        self._instances: dict[str, BaseAgent] = {}
        self._metadata: dict[str, dict] = {}

    def register(self, factory: AgentFactory, metadata: dict | None = None) -> None:
        """Register an agent factory. Call once per agent at startup."""
        instance = factory()
        self._factories[instance.name] = factory
        self._metadata[instance.name] = metadata or {}
        # Don't keep the instance — let GC reclaim until first access

    def get(self, name: str) -> BaseAgent | None:
        """Get an agent instance by name, creating it lazily if needed."""
        if name in self._instances:
            return self._instances[name]
        factory = self._factories.get(name)
        if factory:
            instance = factory()
            self._instances[name] = instance
            return instance
        return None

    def get_by_id(self, agent_id: int) -> BaseAgent | None:
        for metadata in self._metadata.values():
            if metadata.get("agent_id") == agent_id:
                return self.get(metadata["name"])
        return None

    def list_names(self) -> list[str]:
        return sorted(self._factories.keys())

    def list_by_tier(self, tier: int) -> list[str]:
        return sorted(
            name for name, meta in self._metadata.items() if meta.get("tier") == tier
        )

    def list_by_category(self, category: str) -> list[str]:
        return sorted(
            name
            for name, meta in self._metadata.items()
            if meta.get("category") == category
        )

    def get_all(self) -> list[BaseAgent]:
        """Instantiate and return all registered agents."""
        return [self.get(name) for name in self._factories]  # type: ignore[misc]

    def get_performances(self) -> list[AgentPerformance]:
        return [agent.get_performance() for agent in self.get_all()]

    def get_tier_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for meta in self._metadata.values():
            t = meta.get("tier", 0)
            counts[t] = counts.get(t, 0) + 1
        return counts

    def is_registered(self, name: str) -> bool:
        return name in self._factories

    @property
    def agent_count(self) -> int:
        return len(self._factories)


# Global singleton
_registry: AgentRegistry | None = None


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = AgentRegistry()
