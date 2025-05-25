# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py

import json
import math
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal, Optional, Union

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.road.config import RoadVariant

logger = init_logger(__name__)


@dataclass
class PEFTRoadHelper:
    """ 
    A helper class for PEFT configurations, specifically designed for Road.
    This class handles configuration validation, compatibility checks for 
    various LoRA implementations.
    """

    # Required fields
    variant: RoadVariant
    group_size: int
    target_modules: Union[list[str], str]

    modules_to_save: Optional[list[str]] = field(default=None)
    # Extra vllm field, start with 'vllm_' to avoid conflict
    vllm_lora_scaling_factor: float = field(default=1.0)
    vllm_max_position_embeddings: Optional[int] = field(default=False)
    vllm_long_context_scaling_factor: Optional[float] = field(default=None)

    def _validate_features(self) -> list[str]:
        """
        Check if there are any unsupported LoRA features.
        """
        error_msg = []
        if self.modules_to_save:
            error_msg.append("vLLM only supports modules_to_save being None.")
        return error_msg

    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PEFTHelper":
        # Get all field information from the class
        class_fields = {f.name: f for f in fields(cls)}
        # Check for required fields
        required_fields = {
            name
            for name, f in class_fields.items()
            if f.default is MISSING and f.default_factory is MISSING
        }

        # Identify any missing required fields
        missing_fields = required_fields - set(config_dict.keys())
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {missing_fields}")

        # Filter out fields that aren't defined in the class
        filtered_dict = {
            k: v
            for k, v in config_dict.items() if k in class_fields
        }
        return cls(**filtered_dict)

    @classmethod
    def from_local_dir(
            cls,
            lora_path: str,
            max_position_embeddings: Optional[int],
            tensorizer_config_dict: Optional[dict] = None) -> "PEFTHelper":
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")

        if tensorizer_config_dict:
            tensorizer_config = TensorizerConfig(**tensorizer_config_dict)
            tensorizer_args = tensorizer_config._construct_tensorizer_args()
            from tensorizer.stream_io import open_stream
            adapter_config_path = os.path.join(tensorizer_config.lora_dir,
                                            "adapter_config.json")
            with open_stream(adapter_config_path,
                             mode="rb",
                             **tensorizer_args.stream_params) as f:
                config = json.load(f)

            logger.info("Successfully deserialized LoRA config from %s",
                        tensorizer_config.lora_dir)

        else:
            with open(adapter_config_path) as f:
                config = json.load(f)

        config["vllm_max_position_embeddings"] = max_position_embeddings
        return cls.from_dict(config)

    def validate_legal(self, lora_config: LoRAConfig) -> None:
        """
        Validates the LoRA configuration settings against application 
        constraints and requirements.
        """
        error_msg = self._validate_features()
        if error_msg:
            raise ValueError(f"{' '.join(error_msg)}")
