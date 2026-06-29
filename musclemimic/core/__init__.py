"""Musclemimic core module."""

# Import modules to trigger registration of goals, rewards, and termination handlers.
from musclemimic.core import goals  # noqa: F401
from musclemimic.core import initial_state_handler  # noqa: F401
from musclemimic.core.reward import trajectory_based  # noqa: F401
from musclemimic.core.terminal_state_handler import enhanced_bimanual, enhanced_fullbody  # noqa: F401
