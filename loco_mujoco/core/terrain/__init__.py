from .base import Terrain
from .static import StaticTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()

# `boxes` imports `Terrain` from this package, so it has to come after the base classes
# are bound; it registers `BoxTerrain` itself.
from .boxes import BoxSpec, BoxTerrain, TerrainSpec  # noqa: E402
