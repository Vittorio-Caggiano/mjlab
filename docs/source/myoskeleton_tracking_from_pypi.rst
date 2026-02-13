.. _myoskeleton_tracking_from_pypi:

Create and train a custom MyoSkeleton tracking task (PyPI-only)
===============================================================

This tutorial shows how to create your own tracking task based on
``Mjlab-Tracking-Flat-MyoSkeleton`` **without forking mjlab**.
You only install ``mjlab`` from PyPI, then define and register a custom task in
your own repository.

It includes two styles:

1. **Single-file script** (everything in ``train_custom_myo.py``).
2. **Notebook-style workflow** (cells you can paste into Jupyter).

Prerequisites
-------------

- Linux + NVIDIA GPU with working CUDA.
- Python environment with ``pip`` (or ``uv``).
- ``mjlab`` installed from PyPI + ``mujoco-warp`` from GitHub.

Example install:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   pip install git+https://github.com/google-deepmind/mujoco_warp@7c20a44bfed722e6415235792a1b247ea6b6a6d3
   pip install mjlab

Option A) Keep everything in ``train_custom_myo.py``
----------------------------------------------------

Yes — you can keep task registration + training launch in one file.
This is the simplest format for quick experiments.

Create ``train_custom_myo.py``:

.. code-block:: python

   from mjlab.tasks.registry import register_mjlab_task
   from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
   from mjlab.tasks.tracking.config.myoskeleton.env_cfgs import (
     myoskeleton_flat_tracking_env_cfg,
   )
   from mjlab.tasks.tracking.config.myoskeleton.rl_cfg import (
     myoskeleton_tracking_ppo_runner_cfg,
   )
   from mjlab.scripts.train import TrainConfig, launch_training


   CUSTOM_TASK_ID = "Mjlab-Tracking-Flat-MyoSkeleton-Custom"


   def register_task() -> None:
     register_mjlab_task(
       task_id=CUSTOM_TASK_ID,
       env_cfg=myoskeleton_flat_tracking_env_cfg(),
       play_env_cfg=myoskeleton_flat_tracking_env_cfg(play=True),
       rl_cfg=myoskeleton_tracking_ppo_runner_cfg(),
       runner_cls=MotionTrackingOnPolicyRunner,
     )


   if __name__ == "__main__":
     register_task()

     cfg = TrainConfig.from_task(CUSTOM_TASK_ID)

     # Optional: task-specific defaults.
     cfg.agent.experiment_name = "myoskeleton_tracking_custom"
     cfg.agent.run_name = "baseline"
     cfg.agent.max_iterations = 2000
     cfg.agent.logger = "tensorboard"

     # Optional tuning examples.
     # cfg.env.scene.num_envs = 2048
     # cfg.agent.num_steps_per_env = 16
     # cfg.agent.actor.init_noise_std = 0.6

     launch_training(CUSTOM_TASK_ID, cfg)

Run it with a motion file override:

.. code-block:: bash

   python train_custom_myo.py \
     --env.commands.motion.motion-file /absolute/path/to/motion.npz

You can pass more overrides as usual:

.. code-block:: bash

   python train_custom_myo.py \
     --env.commands.motion.motion-file /absolute/path/to/motion.npz \
     --env.scene.num-envs 1024 \
     --agent.algorithm.learning-rate 5e-4 \
     --agent.actor.init-noise-std 0.6

Option B) Notebook-style workflow (Jupyter)
--------------------------------------------

If you prefer a notebook tutorial style, use cells like this.

Cell 1: Imports
^^^^^^^^^^^^^^^

.. code-block:: python

   from mjlab.tasks.registry import register_mjlab_task
   from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
   from mjlab.tasks.tracking.config.myoskeleton.env_cfgs import (
     myoskeleton_flat_tracking_env_cfg,
   )
   from mjlab.tasks.tracking.config.myoskeleton.rl_cfg import (
     myoskeleton_tracking_ppo_runner_cfg,
   )
   from mjlab.scripts.train import TrainConfig, launch_training

Cell 2: Register custom task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   CUSTOM_TASK_ID = "Mjlab-Tracking-Flat-MyoSkeleton-Custom"

   register_mjlab_task(
     task_id=CUSTOM_TASK_ID,
     env_cfg=myoskeleton_flat_tracking_env_cfg(),
     play_env_cfg=myoskeleton_flat_tracking_env_cfg(play=True),
     rl_cfg=myoskeleton_tracking_ppo_runner_cfg(),
     runner_cls=MotionTrackingOnPolicyRunner,
   )

Cell 3: Configure training
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cfg = TrainConfig.from_task(CUSTOM_TASK_ID)
   cfg.agent.experiment_name = "myoskeleton_tracking_custom"
   cfg.agent.run_name = "notebook_baseline"
   cfg.agent.max_iterations = 2000
   cfg.agent.logger = "tensorboard"

Cell 4: Launch training
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   launch_training(
     CUSTOM_TASK_ID,
     cfg,
   )

Then run from terminal (recommended) to pass the motion file override:

.. code-block:: bash

   python train_custom_myo.py \
     --env.commands.motion.motion-file /absolute/path/to/motion.npz

If you do launch directly inside the notebook process, make sure your motion
file is set before training starts (tracking tasks require it).

When to split into multiple files
---------------------------------

Single-file is ideal for fast iteration.
Split into ``my_myo_tracking/tasks.py`` + a launcher later if you want:

- cleaner package structure,
- reusable task registrations,
- easier testing/versioning in larger projects.

Notes
-----

- This workflow keeps your task code independent from the core ``mjlab`` repo.
- You can publish your custom task package separately and version it independently.
- If you use W&B logging in training, set ``cfg.agent.logger = "wandb"`` and
  configure your W&B environment variables.
