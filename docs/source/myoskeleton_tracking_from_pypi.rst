.. _myoskeleton_tracking_from_pypi:

Create and train a custom MyoSkeleton tracking task (PyPI-only)
===============================================================

This tutorial shows how to create your own tracking task based on
``Mjlab-Tracking-Flat-MyoSkeleton`` **without forking mjlab**.
You only install ``mjlab`` from PyPI, then define and register a custom task in
your own repository.

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

Step 1) Create a project for your custom task
----------------------------------------------

.. code-block:: bash

   mkdir my_myo_tracking
   cd my_myo_tracking
   mkdir -p my_myo_tracking
   touch my_myo_tracking/__init__.py

Step 2) Register a custom task in your package
-----------------------------------------------

Create ``my_myo_tracking/tasks.py``:

.. code-block:: python

   from mjlab.tasks.registry import register_mjlab_task
   from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

   from mjlab.tasks.tracking.config.myoskeleton.env_cfgs import (
     myoskeleton_flat_tracking_env_cfg,
   )
   from mjlab.tasks.tracking.config.myoskeleton.rl_cfg import (
     myoskeleton_tracking_ppo_runner_cfg,
   )


   register_mjlab_task(
     task_id="Mjlab-Tracking-Flat-MyoSkeleton-Custom",
     env_cfg=myoskeleton_flat_tracking_env_cfg(),
     play_env_cfg=myoskeleton_flat_tracking_env_cfg(play=True),
     rl_cfg=myoskeleton_tracking_ppo_runner_cfg(),
     runner_cls=MotionTrackingOnPolicyRunner,
   )

This creates a new task ID in your own codebase. From here you can customize:

- environment randomization,
- reward/termination configuration,
- PPO hyperparameters.

Step 3) Add a tiny training launcher
------------------------------------

Create ``train_custom_myo.py`` at project root:

.. code-block:: python

   import my_myo_tracking.tasks  # noqa: F401 (registers custom task)

   from mjlab.scripts.train import TrainConfig, launch_training


   if __name__ == "__main__":
     task_id = "Mjlab-Tracking-Flat-MyoSkeleton-Custom"
     cfg = TrainConfig.from_task(task_id)

     # Optional: task-specific overrides.
     cfg.agent.experiment_name = "myoskeleton_tracking_custom"
     cfg.agent.run_name = "baseline"
     cfg.agent.max_iterations = 2000
     cfg.agent.logger = "tensorboard"

     # You can also tune env and PPO settings here.
     # cfg.env.scene.num_envs = 2048
     # cfg.agent.num_steps_per_env = 16
     # cfg.agent.actor.init_noise_std = 0.6

     launch_training(task_id, cfg)

Step 4) Run training with a motion file
---------------------------------------

Tracking tasks need a motion reference file.
Pass a local ``motion.npz`` with a CLI override:

.. code-block:: bash

   python train_custom_myo.py \
     --env.commands.motion.motion-file /absolute/path/to/motion.npz

You can also pass other overrides the same way, for example:

.. code-block:: bash

   python train_custom_myo.py \
     --env.commands.motion.motion-file /absolute/path/to/motion.npz \
     --env.scene.num-envs 1024 \
     --agent.algorithm.learning-rate 5e-4 \
     --agent.actor.init-noise-std 0.6

Step 5) (Optional) Use the training sequence benchmark script
-------------------------------------------------------------

If you want to run a multi-case sequence with min-iteration tracking:

.. code-block:: bash

   python -m mjlab.scripts.train Mjlab-Tracking-Flat-MyoSkeleton \
     --env.commands.motion.motion-file /absolute/path/to/motion.npz

Or use your local copy of ``run_myoskeleton_training_sequence.py`` and point it
at your motion file.

Notes
-----

- This workflow keeps your task code independent from the core ``mjlab`` repo.
- You can publish your custom task package separately and version it independently.
- If you use W&B logging in training, set ``cfg.agent.logger = "wandb"`` and
  configure your W&B environment variables.
