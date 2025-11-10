# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =========================================================================
# AutoWatermark.py
# Description: This is a generic watermark class that will be instantiated
#              as one of the watermark classes of the library when created
#              with the [`AutoWatermark.load`] class method.
# =========================================================================

import torch
import importlib
from typing import List
from watermark.auto_config import AutoConfig

WATERMARK_MAPPING_NAMES={
    'KGW': 'watermark.kgw.KGW',
    'SynthID': 'watermark.synthid.SynthID',
    'LSH': 'watermark.lsh.LSH',
    'SAM': 'watermark.sam.SAM',
    'EXPEdit': 'watermark.exp_edit.EXPEdit',
    'ITSEdit': 'watermark.its_edit.ITSEdit',
}

def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    if name in WATERMARK_MAPPING_NAMES:
        return WATERMARK_MAPPING_NAMES[name]
    else:
        raise ValueError(f"Invalid algorithm name: {name}")

class AutoWatermark:
    """
        This is a generic watermark class that will be instantiated as one of the watermark classes of the library when
        created with the [`AutoWatermark.load`] class method.

        This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark is designed to be instantiated "
            "using the `AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)` method."
        )

    def load(algorithm_name, algorithm_config=None, transformers_config=None, *args, **kwargs):
        """Load the watermark algorithm instance based on the algorithm name."""
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)
        watermark_config = AutoConfig.load(algorithm_name, transformers_config, algorithm_config_path=algorithm_config, **kwargs)
        watermark_instance = watermark_class(watermark_config)
        return watermark_instance