"""Goals module for musclemimic."""

from musclemimic.core.goals.bimanual import GoalBimanualTrajMimic, GoalBimanualTrajMimicv2
from musclemimic.core.goals.bimanual_tracking import GoalBimanualTracking, GoalBimanualTrackingVisual
from musclemimic.core.goals.fullbody_tracking import GoalFullBodyTracking, GoalFullBodyTrackingVisual
from musclemimic.core.goals.trajectory import GoalTrajMimic, GoalTrajMimicv2

# Register all trajectory mimic goals
GoalTrajMimic.register()
GoalTrajMimicv2.register()
GoalFullBodyTracking.register()
GoalFullBodyTrackingVisual.register()
GoalBimanualTrajMimic.register()
GoalBimanualTrajMimicv2.register()
GoalBimanualTracking.register()
GoalBimanualTrackingVisual.register()

__all__ = [
    "GoalBimanualTrajMimic",
    "GoalBimanualTrajMimicv2",
    "GoalBimanualTracking",
    "GoalBimanualTrackingVisual",
    "GoalFullBodyTracking",
    "GoalFullBodyTrackingVisual",
    "GoalTrajMimic",
    "GoalTrajMimicv2",
]
