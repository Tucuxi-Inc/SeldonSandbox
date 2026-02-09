"""
Extension registry with dependency resolution.

Manages registration, enabling/disabling, and dependency checking
for simulation extensions.
"""

from __future__ import annotations

from typing import Any

from seldon.extensions.base import SimulationExtension


class ExtensionRegistry:
    """
    Manages simulation extensions with dependency resolution.

    Usage::

        registry = ExtensionRegistry()
        registry.register(GeographyExtension())
        registry.register(MigrationExtension(geography_ref))
        registry.enable("geography")
        registry.enable("migration")   # requires geography — checked
    """

    def __init__(self) -> None:
        self._extensions: dict[str, SimulationExtension] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, extension: SimulationExtension) -> None:
        """Register an extension as available (not yet enabled)."""
        self._extensions[extension.name] = extension
        self._enabled[extension.name] = False

    def enable(self, name: str) -> None:
        """Enable a registered extension after checking dependencies."""
        if name not in self._extensions:
            raise KeyError(f"Extension '{name}' is not registered")

        deps = self._extensions[name].get_default_config().get("requires", [])
        for dep in deps:
            if not self._enabled.get(dep, False):
                raise ValueError(
                    f"Extension '{name}' requires '{dep}' to be enabled first"
                )
        self._enabled[name] = True

    def disable(self, name: str) -> None:
        """Disable an extension, checking that no dependents rely on it."""
        if name not in self._extensions:
            raise KeyError(f"Extension '{name}' is not registered")

        # Check reverse dependencies
        for other_name, other_ext in self._extensions.items():
            if self._enabled.get(other_name, False) and other_name != name:
                deps = other_ext.get_default_config().get("requires", [])
                if name in deps:
                    raise ValueError(
                        f"Cannot disable '{name}': "
                        f"extension '{other_name}' depends on it"
                    )
        self._enabled[name] = False

    def is_enabled(self, name: str) -> bool:
        """Check whether a named extension is currently enabled."""
        return self._enabled.get(name, False)

    def get(self, name: str) -> SimulationExtension | None:
        """Return a registered extension by name, or *None*."""
        return self._extensions.get(name)

    def get_enabled(self) -> list[SimulationExtension]:
        """Return all enabled extensions in registration order."""
        return [
            ext for name, ext in self._extensions.items()
            if self._enabled.get(name, False)
        ]

    def get_combined_config(self) -> dict[str, Any]:
        """Return merged default configs for all enabled extensions."""
        return {
            ext.name: ext.get_default_config()
            for ext in self.get_enabled()
        }

    @property
    def registered_names(self) -> list[str]:
        """Names of all registered extensions."""
        return list(self._extensions.keys())

    @property
    def enabled_names(self) -> list[str]:
        """Names of all enabled extensions."""
        return [n for n, enabled in self._enabled.items() if enabled]
