# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. and EPFL. All Rights Reserved.
from typing import Any, Union
from yaml import safe_load, YAMLError
import io
import json
import os
from pathlib import Path
from typing import Optional, Union

from safetensors.torch import load as load_st
from safetensors.torch import save_file

import torch

from hydra.utils import instantiate

__all__ = ["save_safetensors", "load_safetensors", "load_model_from_safetensors"]

# Manual list of allowed _target_ for Hydra instantiation, to avoid any potential 
# issues with untrusted checkpoints. When implementing new targets, please ensure 
# they are added here to make them loadable by Hydra.
ALLOWED_TARGETS = [
    # Wrapper targets
    "flextok_wrapper.FlexTok",
    "flextok_wrapper.FlexTokFromHub",
    "vae_wrapper.StableDiffusionVAE",
    # Flow Matching targets
    "flow_matching.pipelines.MinRFPipeline",
    "flow_matching.noise_modules.MinRFNoiseModule",
    "flow_matching.cfg_utils.MomentumBuffer",
    # Model Utils targets
    "model.utils.wrappers.SequentialModuleDictWrapper",
    "model.utils.posembs.PositionalEmbedding",
    "model.utils.posembs.PositionalEmbeddingAdder",
    "model.utils.dict_ops.DictKeyFilter",
    "model.utils.dict_ops.PerSampleOp",
    "model.utils.dict_ops.PerSampleReducer",
    "model.utils.dict_ops.channels_first_to_last",
    "model.utils.dict_ops.channels_last_to_first",
    # Model Trunks targets
    "model.trunks.transformers.FlexTransformer",
    "model.trunks.transformers.FlexTransformerDecoder",
    # Model Preprocessors targets
    "model.preprocessors.token_dropout.MaskedNestedDropout",
    "model.preprocessors.time_embedding.TimestepEmbedder",
    "model.preprocessors.registers.Registers1D",
    "model.preprocessors.patching.PatchEmbedder",
    "model.preprocessors.nullcond.ZeroNullCond",
    "model.preprocessors.nullcond.LearnedNullCond",
    "model.preprocessors.mask_tokens.MaskTokenModule",
    "model.preprocessors.linear.LinearLayer",
    "model.preprocessors.flex_seq_packing.BlockWiseSequencePacker",
    # Model Postprocessors targets
    "model.postprocessors.seq_unpacking.SequenceUnpacker",
    "model.postprocessors.heads.LinearHead",
    "model.postprocessors.heads.ToPatchesLinearHead",
    "model.postprocessors.heads.MLPHead",
    # Model Layers targets
    "model.layers.transformer_blocks.FlexBlock",
    "model.layers.transformer_blocks.FlexBlockAdaLN",
    "model.layers.transformer_blocks.FlexDecoderBlock",
    "model.layers.transformer_blocks.FlexDecoderBlockAdaLN",
    "model.layers.norm.Fp32LayerNorm",
    "model.layers.mup_readout.MuReadoutFSDP",
    "model.layers.mlp.Mlp",
    "model.layers.mlp.GatedMlp",
    "model.layers.drop_path.DropPath",
    "model.layers.attention.FlexAttention",
    "model.layers.attention.FlexSelfAttention",
    "model.layers.attention.FlexCrossAttention",
    # Regularizers targets
    "regularizers.quantize_fsq.FSQ",
]

MAX_LEN_YAML_PARSE = 10_000


def save_safetensors(state_dict, ckpt_path, metadata_dict=None):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    for k, v in state_dict.items():
        state_dict[k] = v.contiguous()
    if metadata_dict is not None:
        metadata = {k: str(v) for k, v in metadata_dict.items()}
    else:
        metadata = None
    save_file(state_dict, ckpt_path, metadata=metadata)


def _sanitize_hydra_config(cfg: Union[dict, list, tuple]) -> None:
    """
    Recursively scan a Hydra-style config for any `_target_` entries and ensure 
    they only point inside a known list of classes and functions. Raises if any 
    disallowed target is found.
    """
    if isinstance(cfg, dict):
        for key, val in cfg.items():
            if key == "_target_":
                if not isinstance(val, str) or not val in ALLOWED_TARGETS:
                    raise ValueError(f"Potentially unsafe _target_ in Hydra config: {val!r}")
                    pass
            else:
                _sanitize_hydra_config(val)
    elif isinstance(cfg, (list, tuple)):
        for item in cfg:
            _sanitize_hydra_config(item)


def safe_parse_metadata(metadata_str):
    metadata = {}
    for k, v in metadata_str.items():
        if not isinstance(v, str) or len(v) > MAX_LEN_YAML_PARSE:
            metadata[k] = v
            continue
        try:
            parsed = safe_load(v.replace('None', 'null'))
            metadata[k] = parsed
        except YAMLError:
            metadata[k] = v
    return metadata


def load_safetensors(safetensors_path, return_metadata=True):
    with open(safetensors_path, "rb") as f:
        data = f.read()

    tensors = load_st(data)

    if not return_metadata:
        return tensors

    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    metadata = header.get("__metadata__", {})

    # Safely parse and sanitize before handing it off to hydra.utils.instantiate
    metadata = safe_parse_metadata(metadata)
    _sanitize_hydra_config(metadata)
    
    return tensors, metadata


def load_model_from_safetensors(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    to_eval: bool = True,
) -> torch.nn.Module:
    """Loads a safetensors checkpoint from the given path and instantiates
    a model from the config safed in it.

    Args:
        ckpt_path: Path to .safetensors checkpoint
        device: Optional torch device
        to_eval: Set to call .eval() on model

    Returns:
        Model with loaded weights.
    """
    ckpt, config = load_safetensors(ckpt_path)
    model = instantiate(config)
    model.load_state_dict(ckpt)

    if device is not None:
        model = model.to(device)

    if to_eval:
        model = model.eval()

    return model
