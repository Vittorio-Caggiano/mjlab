"""MyoLeg robot config for mjlab velocity task.

Uses submodules/myo_sim/leg/myolegs_mjlab.xml (muscle-controlled, no added actuators).
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlMuscleActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Repo root (parent of src/) for submodules path.
_REPO_ROOT: Path = MJLAB_SRC_PATH.parent.parent
MYOLEGS_MJLAB_XML: Path = (
  _REPO_ROOT / "submodules" / "myo_sim" / "leg" / "myolegs_mjlab.xml"
)
assert MYOLEGS_MJLAB_XML.exists(), f"Missing {MYOLEGS_MJLAB_XML}"


def get_myoleg_spec() -> mujoco.MjSpec:
  """Load the MyoLeg entity-only MjSpec (robot, no terrain)."""
  return mujoco.MjSpec.from_file(str(MYOLEGS_MJLAB_XML))


# Wrap all XML-defined muscle actuators (they target tendons).
MYOLEG_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlMuscleActuatorCfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r".*_tendon",),
    ),
  ),
)

MYOLEG_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype=0,
  conaffinity=1,
  condim=3,
)

# Use keyframe from XML for init state.
MYOLEG_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=None,  # use keyframe from model
  joint_vel={".*": 0.0},
)


def get_myoleg_robot_cfg() -> EntityCfg:
  """Get MyoLeg robot entity config (muscle-controlled, XML actuators only)."""
  return EntityCfg(
    spec_fn=get_myoleg_spec,
    articulation=MYOLEG_ARTICULATION,
    init_state=MYOLEG_INIT_STATE,
    collisions=(MYOLEG_COLLISION,),
  )
