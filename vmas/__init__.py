#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from vmas.interactive_rendering import render_interactively #rendering
from vmas.make_env import make_env #functions to make enviroment
from vmas.simulator.environment import Wrapper #Wrappers used in this paper

from vmas.simulator.utils import _init_pyglet_device #pyglet device

_init_pyglet_device()

__all__ = [ #including all useful modules(make_env, render_interactively...)
    "make_env",
    "render_interactively",
    "Wrapper",
    "scenarios",
    "debug_scenarios",
    "mpe_scenarios",
]

scenarios = sorted(#scenarios
    [
        "dropout",
        "dispersion",
        "transport",
        "reverse_transport",
        "give_way",
        "wheel",
        "balance",
        "football",
        "discovery",
        "flocking",
        "passage",
        "joint_passage_size",
        "joint_passage",
        "ball_passage",
        "ball_trajectory",
        "buzz_wire",
        "multi_give_way",
        "navigation",
    ]
)

debug_scenarios = sorted(#scenarios
    [
        "asym_joint",
        "circle_trajectory",
        "goal",
        "het_mass",
        "line_trajectory",
        "vel_control",
        "waterfall",
    ]
)

mpe_scenarios = sorted(#scenarios in MPE experiments
    [
        "simple",
        "simple_adversary",
        "simple_crypto",
        "simple_push",
        "simple_reference",
        "simple_speaker_listener",
        "simple_spread",
        "simple_tag",
        "simple_world_comm",
    ]
)
