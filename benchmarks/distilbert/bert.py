import torch
import transformers
from torch import device
from typing import Any, Dict, Optional, List, Tuple
from transformers import BertForMaskedLM, BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings, BertOnlyMLMHead, MaskedLMOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from copy import deepcopy

from tspipe.model_wrapper import TSPipeModelWrapper

class Seequential(torch.nn.Sequential):

    def __init__(self, *args: torch.nn.Module, config: Optional[Any] = None):
        super().__init__(*args)
        self.config = config

    def forward(self, **kwargs):
        input = self[0](**kwargs)
        for module in self[1:]:
            input = module(input)
        return input

def bert_to_sequential(model: torch.nn.Module, inputs = None) -> Seequential:
    layers = []
    for child in model.children():
        if isinstance(child, transformers.BertModel):
            inner_layers = _bert_to_sequential(child, model.config)
            layers.extend(inner_layers)
        elif isinstance(child, BertOnlyMLMHead):
            layers.append(BertWrap(child))
    
    sequential_layers = Seequential(*[TSPipeModelWrapper(deepcopy(d)) for d in layers], config = model.config)
    #print(sequential_layers)
    model.eval()
    sequential_layers.eval()

    if inputs is not None:
        # integrity check
        with torch.no_grad():
            orig_output = model(**inputs)
            #print(orig_output)
            seq_output = sequential_layers(**inputs)
            #print(seq_output)
        assert torch.allclose(orig_output.logits, seq_output.logits)

    return sequential_layers

def _bert_to_sequential(module, config):
    layers = []
    for child in module.children():
        if isinstance(child, BertEmbeddings):
            layers.append(BertModelInitWrapper(child, config))
        elif isinstance(child, BertEncoder):
            layers.append(BertFisrBlock(child.layer[0], config.add_cross_attention))
            for layer in child.layer[1:-1]:
                layers.append(BertIntermediateBlock(layer))
            layers.append(BertLastBlock(child.layer[-1], config))
        else:
            layers.append(child)
    return layers
class BertWrap(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.cls = head
    
    def forward(self, encoder_outputs):
        return_dict = not isinstance(encoder_outputs, tuple)

        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        else:
            outputs = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = None
        ###LABELS WILL BE GIVEN
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        else:
            return MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

class BertLastBlock(torch.nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.layer = layer

    def forward(self, inputs):
        hidden_states, attention_mask, head_mask, encoder_hidden_states, \
        encoder_attention_mask, past_key_values, output_attentions, \
        index, all_hidden_states, all_self_attentions, all_cross_attentions, \
        next_decoder_cache, output_hidden_states, use_cache, add_cross_attention, \
        return_dict = inputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[index] if head_mask is not None else None
        past_key_value = past_key_values[index] if past_key_values is not None else None

        layer_outputs = self.layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )


class BertIntermediateBlock(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, inputs):
        hidden_states, attention_mask, head_mask, encoder_hidden_states, \
        encoder_attention_mask, past_key_values, output_attentions, \
        index, all_hidden_states, all_self_attentions, all_cross_attentions, \
        next_decoder_cache, output_hidden_states, use_cache, add_cross_attention, \
        return_dict = inputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[index] if head_mask is not None else None
        past_key_value = past_key_values[index] if past_key_values is not None else None

        layer_outputs = self.layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # for hidden states, only keep the last encoder output
        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states[-1:]
        
        next_block_input = [hidden_states, attention_mask, head_mask, encoder_hidden_states, 
                            encoder_attention_mask, past_key_values, output_attentions, 
                            index + 1, all_hidden_states, all_self_attentions, all_cross_attentions,
                            next_decoder_cache, output_hidden_states, use_cache, add_cross_attention,
                            return_dict]
        
        return next_block_input
        
class BertFisrBlock(torch.nn.Module):
    def __init__(self, layer, add_cross_attention):
        super().__init__()
        self.layer = layer
        self.add_cross_attention = add_cross_attention
    
    def forward(self, inputs):

        hidden_states, attention_mask, head_mask, encoder_hidden_states, \
        encoder_attention_mask, past_key_values, use_cache, output_attentions, \
        output_hidden_states, return_dict = inputs

        index = 0
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[index] if head_mask is not None else None
        past_key_value = past_key_values[index] if past_key_values is not None else None

        layer_outputs = self.layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        next_block_input = [hidden_states, attention_mask, head_mask, encoder_hidden_states, 
                            encoder_attention_mask, past_key_values, output_attentions, 
                            index + 1, all_hidden_states, all_self_attentions, all_cross_attentions,
                            next_decoder_cache, output_hidden_states, use_cache, self.add_cross_attention,
                            return_dict]
        
        return next_block_input

class BertModelInitWrapper(torch.nn.Module):
    def __init__(self, embedding, config):
        super().__init__()
        self.model = BertModelInit(embedding, config)
    
    def forward(self, input: Dict):
        return self.model(**input)
    
class BertModelInit(torch.nn.Module):
    def __init__(self, embedding, config):
        super().__init__()
        self.embeddings = embedding
        self.config= config
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_input = [embedding_output, extended_attention_mask, head_mask,
                        encoder_hidden_states, encoder_extended_attention_mask,
                        past_key_values, use_cache, output_attentions, output_hidden_states, 
                        return_dict]
        
        return encoder_input

    def get_head_mask(
        self, head_mask: Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> torch.Tensor:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: device = None
    ) -> torch.Tensor:
        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        self.dtype = next(self.embeddings.parameters()).dtype
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

def main():
    configuration = BertConfig(num_hidden_layers=3)
    teacher_model = BertForMaskedLM(configuration)
    print(teacher_model)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print(inputs)
    sequential_layers = bert_to_sequential(teacher_model, inputs)
    
if __name__ == '__main__':
    main()