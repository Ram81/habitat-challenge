#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
import os
import contextlib

import numpy as np
import torch
from gym.spaces import Discrete, Dict, Box

import habitat
from habitat import Config, logger
from habitat.core.agent import Agent

from src import POLICY_CLASSES
from src.default import get_config
from src.models.common import batch_obs


class ILAgent(Agent):
    def __init__(self, config: Config):
        if not config.MODEL_PATH:
            raise Exception(
                "Model checkpoint wasn't provided, quitting."
            )
        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        # print("torch: {}".format(torch.__version__))
        # print(config.MODEL_PATH)
        # print(os.listdir("habitat-challenge-data"))
        # print(os.listdir("habitat-challenge-data/data/"))
        # print(os.listdir("habitat-challenge-data/data/scene_datasets/"))
        ckpt_dict = torch.load(config.MODEL_PATH, map_location=self.device)

        # Config
        self.config = config
        config = self.config.clone()
        self.model_cfg = config.MODEL
        il_cfg = config.IL.BehaviorCloning
        task_cfg = config.TASK_CONFIG.TASK

        # Load spaces (manually)
        spaces = {
            "objectgoal": Box(
                low=0, high=20, # from matterport dataset
                shape=(1,),
                dtype=np.int64
            ),
            "depth": Box(
                low=0,
                high=1,
                shape=(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1),
                dtype=np.float32,
            ),
            "rgb": Box(
                low=0,
                high=255,
                shape=(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3),
                dtype=np.uint8,
            ),
            "gps": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,), # Spoof for model to be shaped correctly
                dtype=np.float32,
            ),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)
        }

        observation_spaces = Dict(spaces)
        if 'action_distribution.linear.bias' in ckpt_dict:
            num_acts = ckpt_dict['action_distribution.linear.bias'].size(0)
        action_spaces = Discrete(num_acts)

        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        policy_class = POLICY_CLASSES[il_cfg.POLICY.name]
        self.model = policy_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            model_config=self.model_cfg,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            additional_sensors=additional_sensors,
        ).to(self.device)

        self.model.load_state_dict(ckpt_dict)
        self.model.eval()

        self.semantic_predictor = None

        # Load other items
        self.test_recurrent_hidden_states = torch.zeros(
            self.model_cfg.STATE_ENCODER.num_recurrent_layers,
            1, # num_processes
            self.model_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        self.not_done_masks = None
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

        # self.step = 0
        self.ep = 0
        # self._POSSIBLE_ACTIONS = task_cfg.POSSIBLE_ACTIONS
        # self.actions = []

    def reset(self):
        # print(f'{self.ep} reset {self.step}')
        # We don't reset state because our rnn accounts for masks, and ignore actions because we don't use actions
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.test_recurrent_hidden_states = torch.zeros(
            self.model_cfg.STATE_ENCODER.num_recurrent_layers,
            1, # num_processes
            self.model_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        # self.step = 0
        self.ep += 1
        logger.info("Episode done: {}".format(self.ep))
        # self.actions = []

    @torch.no_grad()
    def act(self, observations):
        batch = batch_obs([observations], device=self.device) # Why is this put in a list?
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"]) - 1

        with torch.no_grad():
            # Substitute 3D GPS (2D provided noted in `nav.py`)
            # if batch['gps'].size(-1) == 2:
            #     batch["gps"] = torch.stack([
            #         batch["gps"][:, 1],
            #         torch.zeros(batch["gps"].size(0), dtype=batch["gps"].dtype, device=self.device),
            #         -batch["gps"][:, 0],
            #     ], axis=-1)

            logits, self.test_recurrent_hidden_states = self.model(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            actions = torch.argmax(logits, dim=1)
            self.prev_actions.copy_(actions)

        self.not_done_masks = torch.ones(1, 1, device=self.device, dtype=torch.bool) # Reset called externally, we're not done until then
        return actions[0].item()


def main():
    # ! Note, there's some additional config not ported from dev setup, but those choices shouldn't matter...
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--config-path", type=str, required=True, default="configs/aux_objectnav.yaml")
    args = parser.parse_args()

    DEFAULT_CONFIG = "configs/il_objectnav.yaml"
    config = get_config([DEFAULT_CONFIG, args.config_path],
                ['BASE_TASK_CONFIG_PATH', config_paths]).clone()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL_PATH = args.model_path

    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    config.RANDOM_SEED = 7
    config.freeze()
    torch.backends.cudnn.benchmark = False

    agent = ILAgent(config)
    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()