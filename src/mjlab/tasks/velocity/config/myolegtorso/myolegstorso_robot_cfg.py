"""MyoLegTorso robot config for mjlab velocity task.

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
MYOLEGSSTORSO_MJLAB_XML: Path = (
  _REPO_ROOT / "submodules" / "myo_sim" / "leg" / "myolegstorso_mjlab.xml"
)
assert MYOLEGSSTORSO_MJLAB_XML.exists(), f"Missing {MYOLEGSSTORSO_MJLAB_XML}"


def get_myolegstorso_spec() -> mujoco.MjSpec:
  """Load the MyoLeg entity-only MjSpec (robot, no terrain)."""
  spec = mujoco.MjSpec.from_file(str(MYOLEGSSTORSO_MJLAB_XML))
  # Force SPARSE Jacobian to trigger the correct branch in mujoco_warp/mjlab
  # This avoids a reshape ValueError for models with many tendons.
  spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  return spec


# Wrap all XML-defined muscle actuators (they target tendons).
MYOLEGSTORSO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlMuscleActuatorCfg(
      transmission_type=TransmissionType.TENDON,
      target_names_expr=(r".*_tendon",),
    ),
  ),
)

MYOLEGSTORSO_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype=0,
  conaffinity=1,
  condim=3,
)

# Use keyframe from XML for init state.
MYOLEGSTORSO_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=None,  # use keyframe from model
  joint_vel={".*": 0.0},
)


def get_myolegstorso_robot_cfg() -> EntityCfg:
  """Get MyoLeg robot entity config (muscle-controlled, XML actuators only)."""
  return EntityCfg(
    spec_fn=get_myolegstorso_spec,
    articulation=MYOLEGSTORSO_ARTICULATION,
    init_state=MYOLEGSTORSO_INIT_STATE,
    collisions=(MYOLEGSTORSO_COLLISION,),
  )
