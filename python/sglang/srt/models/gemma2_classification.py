# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

from typing import Iterable, Optional, Tuple, Set

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma2 import Gemma2ForCausalLM, Gemma2Model


class LMSYSHead(nn.Module):
    def __init__(self, in_features, num_labels=3):
        super().__init__()
        self.head = nn.Linear(in_features, num_labels, bias=False)

    def forward(self, hidden_states, **kwargs):
        logits = self.head(hidden_states)
        return logits


class CustomGemma2ForClassification(nn.Module):
    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma2Model(config, quant_config=quant_config)

        self.classification_head = LMSYSHead(in_features=config.hidden_size, num_labels=config.num_labels)

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            input_embeds: torch.Tensor = None,
            get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "CustomGemma2ForClassification is only used for embedding. Please add --is-embedding when you launch the server."

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        last_token_hidden = self.pooler(hidden_states, forward_batch).embeddings
        scores = self.classification_head(last_token_hidden)

        return EmbeddingPoolerOutput(scores)

    # def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    #     params_dict = dict(self.named_parameters())
    #
    #     for name, loaded_weight in weights:
    #         if "classification_head" in name:
    #             param = params_dict[name]
    #             weight_loader = getattr(param, "weight_loader", default_weight_loader)
    #             weight_loader(param, loaded_weight)
    #         elif "lm_head" in name:
    #             continue
    #         else:
    #             Gemma2ForCausalLM.load_weights(self, [(name, loaded_weight)])

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                "Some weights are not initialized from checkpoints: "
                f"{unloaded_params}"
            )


EntryClass = CustomGemma2ForClassification
