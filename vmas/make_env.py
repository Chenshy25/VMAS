#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Optional

from vmas import scenarios
from vmas.simulator.environment import Environment
from vmas.simulator.environment import Wrapper


def make_env( #this is the function used to make environments from local 'scenarios' file folder
    scenario_name,
    num_envs: int = 32,
    device: str = "cpu",
    continuous_actions: bool = True,
    wrapper: Optional[
        Wrapper
    ] = None,  # One of: None, vmas.Wrapper.RLLIB, and vmas.Wrapper.GYM
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs,
):
    # load scenario from script
    if not scenario_name.endswith(".py"):
        scenario_name += ".py"
    scenario = scenarios.load(scenario_name).Scenario()
    env = Environment(
        scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        seed=seed,
        **kwargs,
    )

    return wrapper.get_env(env) if wrapper is not None else env
