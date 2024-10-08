from typing import List, Tuple, Optional
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3Model, Phi3DecoderLayer, Phi3MLP, Phi3Config, Cache, DynamicCache, logger
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from torch.nn import CrossEntropyLoss, Module, Sigmoid, ModuleList, Linear, BCEWithLogitsLoss
import torch
import re
from dataclasses import dataclass
from huggingface_hub import PyTorchModelHubMixin

@dataclass
class Phi3exModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    ar_loss: Optional[torch.FloatTensor] = None
    gating_loss: Optional[torch.FloatTensor] = None
    gating_output: Optional[torch.FloatTensor] = None

class Gate(Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.gate = Linear(self.hidden_dim, self.num_experts, bias=False)
        self.sig_func = Sigmoid()
        self.threshold = config.threshold

    def forward(self, cls_hidden_states: Tensor) -> tuple[Tensor]:
        gating_logits = self.gate(cls_hidden_states)
        gating_output = self.sig_func(gating_logits)
        gating_output = torch.where(gating_output > self.threshold, 1, 0)
        return gating_logits, gating_output

class ExpertsModule(Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts = ModuleList([Phi3MLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: Tensor, expert_indices: Tensor) -> Tensor:
        outputs = torch.zeros_like(hidden_states)
        # shape_index = [i for i in range(len(expert_indices.shape))]
        # shape_index[-1], shape_index[-2] = shape_index[-2], shape_index[-1]
        # expert_indices = torch.permute(expert_indices, tuple(shape_index))
        expert_indices = expert_indices.permute(1, 0)
        for i, expert_index_tensor in enumerate(expert_indices):
            ones_indices = torch.nonzero(expert_index_tensor)
            expert_batch = torch.index_select(hidden_states, 0, ones_indices.view(-1))
            output = self.experts[i](expert_batch)
            outputs.index_add_(0, ones_indices.view(-1), output)

        return outputs

class Phi3exDecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.mlp
        self.mlp = ExpertsModule(config)

    def forward(self, 
                hidden_states: Tensor, 
                attention_mask: Tensor | None = None, 
                position_ids: LongTensor | None = None, 
                past_key_value: Tuple[Tensor] | None = None, 
                output_attentions: bool | None = False, 
                use_cache: bool | None = False,
                expert_indices: Tensor | None = None) -> Tuple[FloatTensor, Tuple[FloatTensor, FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, expert_indices)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Phi3exModel(Phi3Model):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        del self.layers
        self.layers = nn.ModuleList(
            [Phi3exDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(self, 
                input_ids: LongTensor = None, 
                attention_mask: Tensor | None = None, 
                position_ids: LongTensor | None = None, 
                past_key_values: List[FloatTensor] | None = None, 
                inputs_embeds: FloatTensor | None = None, 
                use_cache: bool | None = None, 
                output_attentions: bool | None = None, 
                output_hidden_states: bool | None = None, 
                return_dict: bool | None = None,
                expert_indices: Tensor | None = None) -> Tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        attention_mask[..., 0] = torch.finfo(inputs_embeds.dtype).min
        if(attention_mask.shape[-1] == attention_mask.shape[-2]):
            attention_mask[..., 0, :] = torch.zeros_like(attention_mask[0, 0, 0])

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    expert_indices
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    expert_indices=expert_indices
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Phi3exForCausalLM(Phi3ForCausalLM, PyTorchModelHubMixin):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        del self.model
        self.model = Phi3exModel(config)
        self.gating_model = Gate(config)
        self.loss_fct = CrossEntropyLoss()
        self.gating_loss_fct = BCEWithLogitsLoss()

    def forward(self, 
                input_ids: LongTensor = None, 
                attention_mask: Tensor | None = None, 
                position_ids: LongTensor | None = None, 
                past_key_values: List[FloatTensor] | None = None, 
                inputs_embeds: FloatTensor | None = None, 
                labels: LongTensor | None = None, 
                use_cache: bool | None = None, 
                output_attentions: bool | None = None, 
                output_hidden_states: bool | None = None, 
                return_dict: bool | None = None,
                expert_indices: Tensor | None = None,
                compute_gating: bool | None = True) -> Tuple | Phi3exModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        expert_indices = expert_indices.type(torch.FloatTensor)
        expert_indices = expert_indices.to(torch.device(0))

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            expert_indices=expert_indices
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if(compute_gating):
            gating_logits, gating_output = self.gating_model(hidden_states[..., 0, :])
            gating_loss = self.gating_loss_fct(gating_logits, expert_indices)
        else:
            gating_output = None
            gating_loss = None

        ar_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ar_loss = self.loss_fct(shift_logits, shift_labels)

        if(gating_loss is not None and ar_loss is not None):
            loss = self.config.gating_loss_weight * gating_loss + self.config.ar_loss_weight * ar_loss
        elif(gating_loss is not None and ar_loss is None):
            loss = gating_loss
        elif(ar_loss is not None and gating_loss is None):
            loss = ar_loss
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Phi3exModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ar_loss=ar_loss,
            gating_loss=gating_loss,
            gating_output=gating_output
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "expert_indices": kwargs.get("expert_indices", None),
                "compute_gating": kwargs.get("compute_gating", False)
            }
        )
        return model_inputs
    
def transfer_phi3_weights(model: Phi3ForCausalLM, model_new: Phi3exForCausalLM, num_experts: int) -> Phi3exForCausalLM:
    """Transfers weights from Phi3ForCausalLM to Phi3exForCausalLM. Specifically copies weights directly except for mlp layer where first all experts are
    created and each is transferred the original mlp weights. First the mlp.gate_up_proj and down_proj are identified. Once identified the
    mlp.identified_proj_layer is replaced with mlp.experts.expert_id.identified_proj_layer and then its assigned the original weights.

    Args:
        model (Phi3ForCausalLM): The pretrained Phi3 model for CausalLM whose weights we want to transfer
        model_new (Phi3exForCausalLM): The new experts model where we want to transfer those pretrained weights
        num_experts (int): The number of experts in the new Phi3ex model.

    Returns:
        Phi3exForCausalLM: The Phi3ex model with the pretrained weights loaded
    """

    weights_dict = model.state_dict()
    replacement_holder = "mlp.experts.%s"
    for key in list(weights_dict):
        searched_op = re.search(r"(model\.layers\.\d{1,4}\.mlp\.gate_up_proj\.weight)|(model\.layers\.\d{1,4}\.mlp\.down_proj\.weight)", key)
        if not searched_op:
            continue

        mlp_type = searched_op.group()
        weights = weights_dict.pop(mlp_type)

        for i in range(num_experts):
            replace_layer = re.sub(r"mlp", replacement_holder % i, mlp_type)
            weights_dict[replace_layer] = weights

    weights_dict["gating_model.gate.weight"] = model_new.state_dict()["gating_model.gate.weight"]

    model_new.load_state_dict(weights_dict)
    return model_new