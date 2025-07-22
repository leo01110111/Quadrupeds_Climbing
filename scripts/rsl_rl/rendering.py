from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Callable, List, SupportsFloat

import numpy as np
import torch

import gymnasium as gym
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.error import DependencyNotInstalled


__all__ = [
    "RenderCollection",
    "RecordVideo",
    "HumanRendering",
]


class RenderCollection(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Collect rendered frames of an environment such `render` returns a `list[RenderedFrame]`.

    No vector version of the wrapper exists.
    """
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        pop_frames: bool = True,
        reset_clean: bool = True,
    ):
        """Initialize a :class:`RenderCollection` instance.

        Args:
            env: The environment that is being wrapped
            pop_frames (bool): If true, clear the collection frames after `meth:render` is called. Default value is `True`.
            reset_clean (bool): If true, clear the collection frames when `meth:reset` is called. Default value is `True`.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, pop_frames=pop_frames, reset_clean=reset_clean
        )
        gym.Wrapper.__init__(self, env)

        assert env.render_mode is not None
        assert not env.render_mode.endswith("_list")

        self.frame_list: list[RenderFrame] = []
        self.pop_frames = pop_frames
        self.reset_clean = reset_clean

        self.metadata = deepcopy(self.env.metadata)
        if f"{self.env.render_mode}_list" not in self.metadata["render_modes"]:
            self.metadata["render_modes"].append(f"{self.env.render_mode}_list")

    @property
    def render_mode(self):
        """Returns the collection render_mode name."""
        return f"{self.env.render_mode}_list"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform a step in the base environment and collect a frame."""
        output = super().step(action)
        self.frame_list.append(super().render())
        return output

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment, eventually clear the frame_list, and collect a frame."""
        output = super().reset(seed=seed, options=options)

        if self.reset_clean:
            self.frame_list = []
        self.frame_list.append(super().render())

        return output

    def render(self) -> list[RenderFrame]:
        """Returns the collection of frames and, if pop_frames = True, clears it."""
        frames = self.frame_list
        if self.pop_frames:
            self.frame_list = []

        return frames


class RecordVideo(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Records videos of the best-performing robot in a vectorized environment.

    Usually, you only want to record episodes intermittently, say every hundredth episode or at every thousandth environment step.
    To do this, you can specify `episode_trigger` or `step_trigger`.
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.

    The `episode_trigger` should return `True` on the episode when recording should start.
    The `step_trigger` should return `True` on the n-th environment step that the recording should be started, where n sums over all previous episodes.
    If neither `episode_trigger` nor `step_trigger` is passed, a default `episode_trigger` will be employed, i.e. `capped_cubic_video_schedule`.
    This function starts a video at every episode that is a power of 3 until 1000 and then every 1000 episodes.
    By default, the recording will be stopped once reset is called.
    However, you can also create recordings of fixed length (possibly spanning several episodes)
    by passing a strictly positive value for `video_length`.

    No vector version of the wrapper exists.
    """
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        track_best_robot: bool = False,
    ):
        """Wrapper records videos of the best-performing robot based on cumulative rewards.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns `True` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns `True` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            fps (int): The frame per second in the video. Provides a custom video fps for environment, if `None` then
                the environment metadata `render_fps` key is used if it exists, otherwise a default value of 30 is used.
            disable_logger (bool): Whether to disable moviepy logger or not, default it is disabled
            track_best_robot (bool): If True, records only the best-performing robot based on cumulative rewards
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
            track_best_robot=track_best_robot,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo. "
                "Initialize your environment with a render_mode that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule
            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger
        self.track_best_robot = track_best_robot

        self.video_folder = os.path.abspath(video_folder)
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = fps
        self.name_prefix: str = name_prefix
        self._video_name: str | None = None
        self.video_length: int = video_length if video_length != 0 else float("inf")
        self.recording: bool = False
        self.recorded_frames: list[RenderFrame] = []
        self.render_history: list[RenderFrame] = []

        self.step_id = -1
        self.episode_id = -1

        # Initialize reward tracking for vectorized environments
        if self.track_best_robot:
            try:
                self.num_envs = self.env.unwrapped.num_envs
                self.device = self.env.unwrapped.device
            except AttributeError:
                raise ValueError("Environment must support vectorized operations with 'num_envs' and 'device' attributes for tracking best robot.")
            self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.best_env_idx = 0

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "gymnasium[other]"`'
            ) from e

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, List):
            if len(frame) == 0:  # render was called
                return
            self.render_history += frame
            frame = frame[-1]

        if self.track_best_robot:
            if not isinstance(frame, np.ndarray) or frame.ndim != 4 or frame.shape[0] != self.num_envs:
                self.stop_recording()
                logger.warn(
                    f"Recording stopped: expected frame to be a 4D numpy array with shape[0] == num_envs ({self.num_envs}), got {type(frame)} with shape {getattr(frame, 'shape', None)}."
                )
                return
            # Capture frame only for the best-performing environment
            self.recorded_frames.append(frame[self.best_env_idx])
        else:
            if isinstance(frame, np.ndarray):
                self.recorded_frames.append(frame)
            else:
                self.stop_recording()
                logger.warn(
                    f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
                )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")
        if self.recording:
            self._capture_frame()
            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        if self.track_best_robot:
            self.cur_reward_sum.fill_(0.0)
            self.best_env_idx = np.random.choice(self.num_envs)  # Random initial choice

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.track_best_robot:
            self.cur_reward_sum += torch.as_tensor(rew, device=self.device)
            if torch.any(terminated | truncated):
                new_ids = (terminated | truncated).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    self.best_env_idx = torch.argmax(self.cur_reward_sum).item()
                    self.cur_reward_sum[new_ids] = 0

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self.recording:
            self._capture_frame()

            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame]:
        """Compute the render frames as specified by render_mode attribute during initialization of the environment."""
        render_out = super().render()
        if self.recording and isinstance(render_out, List):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            return tmp_history + render_out
        else:
            return render_out

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:
            self.stop_recording()

        self.recording = True
        self._video_name = video_name

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(path, logger=moviepy_logger)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

    def __del__(self):
        """Warn the user in case last video wasn't saved."""
        if len(self.recorded_frames) > 0:
            logger.warn("Unable to save last video! Did you call close()?")


class HumanRendering(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Allows human like rendering for environments that support "rgb_array" rendering."""
    ACCEPTED_RENDER_MODES = [
        "rgb_array",
        "rgb_array_list",
        "depth_array",
        "depth_array_list",
    ]

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Initialize a :class:`HumanRendering` instance.

        Args:
            env: The environment that is being wrapped
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        self.screen_size = None
        self.window = None
        self.clock = None

        assert (
            self.env.render_mode in self.ACCEPTED_RENDER_MODES
        ), f"Expected env.render_mode to be one of {self.ACCEPTED_RENDER_MODES} but got '{env.render_mode}'"
        assert (
            "render_fps" in self.env.metadata
        ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

        if "human" not in self.metadata["render_modes"]:
            self.metadata = deepcopy(self.env.metadata)
            self.metadata["render_modes"].append("human")

    @property
    def render_mode(self):
        """Always returns `'human'`."""
        return "human"

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        """Perform a step in the base environment and render a frame to the screen."""
        result = super().step(action)
        self._render_frame()
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment and render a frame to the screen."""
        result = super().reset(seed=seed, options=options)
        self._render_frame()
        return result

    def render(self) -> None:
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def _render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            )
        assert self.env.render_mode is not None
        if self.env.render_mode.endswith("_list"):
            last_rgb_array = self.env.render()
            assert isinstance(last_rgb_array, list)
            last_rgb_array = last_rgb_array[-1]
        else:
            last_rgb_array = self.env.render()

        assert isinstance(
            last_rgb_array, np.ndarray
        ), f"Expected `env.render()` to return a numpy array, actually returned {type(last_rgb_array)}"

        rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))

        if self.screen_size is None:
            self.screen_size = rgb_array.shape[:2]

        assert (
            self.screen_size == rgb_array.shape[:2]
        ), f"The shape of the rgb array has changed from {self.screen_size} to {rgb_array.shape[:2]}"

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        """Close the rendering window."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
        super().close()