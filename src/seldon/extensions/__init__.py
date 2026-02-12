"""Optional simulation extensions."""

from seldon.extensions.base import SimulationExtension
from seldon.extensions.registry import ExtensionRegistry
from seldon.extensions.geography import GeographyExtension, Location
from seldon.extensions.migration import MigrationExtension
from seldon.extensions.resources import ResourcesExtension
from seldon.extensions.technology import TechnologyExtension
from seldon.extensions.culture import CultureExtension
from seldon.extensions.conflict import ConflictExtension
from seldon.extensions.social_dynamics import SocialDynamicsExtension
from seldon.extensions.diplomacy import DiplomacyExtension
from seldon.extensions.economics import EconomicsExtension
from seldon.extensions.environment import EnvironmentExtension
from seldon.extensions.epistemology import EpistemologyExtension

__all__ = [
    "SimulationExtension",
    "ExtensionRegistry",
    "GeographyExtension",
    "Location",
    "MigrationExtension",
    "ResourcesExtension",
    "TechnologyExtension",
    "CultureExtension",
    "ConflictExtension",
    "SocialDynamicsExtension",
    "DiplomacyExtension",
    "EconomicsExtension",
    "EnvironmentExtension",
    "EpistemologyExtension",
]
