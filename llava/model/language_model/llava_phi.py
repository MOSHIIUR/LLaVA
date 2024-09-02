#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import wandb
from ..load_balancing_loss import *


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                            PhiConfig, PhiForCausalLM, PhiModel
# from .phi.configuration_phi import PhiConfig
# from .phi.modeling_phi import PhiModel, PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.distributed as dist


class LlavaPhiConfig(PhiConfig):
    model_type = "llava_phi"


class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = LlavaPhiConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.gate_logits = None
        self.gate_logits = [] # tuple of gate logits for each steps
        self.gate_logits_encoder = [] # tuple of gate logits for each steps
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import ipdb
        # ipdb.set_trace()
        # print(f'rank {dist.get_rank()}', 'before prepare_inputs_labels_for_multimodal')
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                gate_logits,
                alignment_loss,
                gate_logits_encoder
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

            if gate_logits is not None:
                self.gate_logits.append(gate_logits.cpu().detach())

            if gate_logits_encoder is not None:
                self.gate_logits_encoder.append(gate_logits_encoder)

        # self.gate_logits = (gate_logits,) # tuple of gate logits for each layer
        # self.gate_logits = gate_logits # tuple of gate logits for each layer
        # self.all_gate_logits += (gate_logits,) # tuple of gate logits for each layer
        # self.constrastive_loss = C_loss


        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        if self.config.training:
            projector_type = getattr(self.config, 'mm_projector_type', 'linear')
            encoder_moe_loss = None
            alignment_loss = alignment_loss*self.config.clip_loss_coef
            
            if projector_type == 'sparse_moe':
                load_balancing_loss = aux_loss(
                gate_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                ) * self.config.aux_loss_coef

                if gate_logits_encoder is not None:
                    encoder_moe_loss = load_balancing_loss_func(
                        gate_logits_encoder, 
                        self.config.num_experts,
                        self.config.num_experts_per_tok,
                        )* self.config.aux_loss_coef



                if encoder_moe_loss is None:
                    llm_loss = out['loss']
                    out['loss'] = llm_loss + load_balancing_loss.to(llm_loss.device) + alignment_loss.to(llm_loss.device)


                    if self.config.local_rank == 0:
                        print(f'LLM Loss: {llm_loss}; LoadBalancingLoss: {load_balancing_loss}; AlignmentLoss: {alignment_loss}')
                        print(f'Total Loss: {out["loss"]}')

                        wandb.log({
                        "llm_loss": llm_loss,
                        "load_balancing_loss": load_balancing_loss,
                        "alignment_loss": alignment_loss,
                        })

                else:
                    llm_loss = out['loss']
                    out['loss'] = llm_loss + load_balancing_loss.to(llm_loss.device) + encoder_moe_loss.to(llm_loss) + alignment_loss.to(llm_loss.device)


                    if self.config.local_rank == 0:
                        print(f'LLM Loss: {llm_loss}; LoadBalancingLoss: {load_balancing_loss}; Encoder_moe_loss: {encoder_moe_loss}; AlignmentLoss: {alignment_loss}')
                        print(f'Total Loss: {out["loss"]}')

                        wandb.log({
                        "llm_loss": llm_loss,
                        "load_balancing_loss": load_balancing_loss,
                        "alignment_loss": alignment_loss,
                        "encoder_load_balancing_loss": encoder_moe_loss,
                        })



            else:
                llm_loss = out['loss']
                out['loss'] = llm_loss + alignment_loss.to(llm_loss.device)


                if self.config.local_rank == 0:
                    print(f'LLM Loss: {llm_loss}; AlignmentLoss: {alignment_loss}')
                    print(f'Total Loss: {out["loss"]}')

                    wandb.log({
                        "llm_loss": llm_loss,
                        "alignment_loss": alignment_loss,
                        # ... log any other metrics you want (e.g., accuracy) ... 
                    })
            

        return out


    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
    

AutoConfig.register("llava_phi", LlavaPhiConfig)
# AutoTokenizer.register(LlavaPhiConfig, PhiTokenizer)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
