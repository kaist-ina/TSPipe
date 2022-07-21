from copy import deepcopy
from operator import mod
from statistics import mode
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from torch import Tensor, device
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.models.distilbert.modeling_distilbert import (Embeddings,
                                                                Transformer)

from bert import Seequential
from tspipe.model_wrapper import TSPipeModelWrapper


def ditilbert_to_sequential(model: torch.nn.Module, inputs = None) -> Seequential:
    layers = []
    model_children = list(model.children())
    layers.extend(_ditilbert_to_sequential(model_children[1]))
    layers.append(DistilBertModelWrap(model_children[0], model_children[2], 
                            model_children[3], model_children[4]))

    sequential_layers = Seequential(*[TSPipeModelWrapper(deepcopy(d)) for d in layers], config = model.config)
    
    if inputs:
        with torch.no_grad():
            orig_output = model(**inputs)
            print(orig_output)
            seq_output = sequential_layers(**inputs)
            print(seq_output)
        assert torch.allclose(orig_output.logits, seq_output.logits)
    return sequential_layers

def _ditilbert_to_sequential(module):
    layers = []
    for child in module.children():
        if isinstance(child, Embeddings):
            layers.append(DistilBertModelInitWrapper(child, module.config))
        elif isinstance(child, Transformer):
            layers.append(TransformerFirstBlock(child.layer[0]))
            for layer in child.layer[1:-1]:
                layers.append(TransforemrIntermediateBlock(layer))
            layers.append(TransformerLastBlock(child.layer[-1]))
        else:
            layers.append(child)
    return layers

class DistilBertModelWrap(torch.nn.Module):
    def __init__(self, activation, vocab_transform, vocab_layer_norm, vocab_projector) -> None:
        super().__init__()
        self.vocab_transform = vocab_transform
        self.activation = activation
        self.vocab_layer_norm = vocab_layer_norm
        self.vocab_projector = vocab_projector
        self.mlm_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):

        return_dict = not isinstance(inputs, tuple)
        hidden_states = inputs[0]

        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        # if labels is not None:
        #     mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + inputs[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=inputs.hidden_states[-1:],
            attentions=inputs.attentions,
        )


class TransformerLastBlock(torch.nn.Module):
    def __init__(self, block) -> None:
        super().__init__()
        self.layer = block

    def forward(self, inputs):
        hidden_state, attn_mask, head_mask, \
        output_attentions, output_hidden_states, return_dict, \
        all_hidden_states, all_attentions, index = inputs
        
        if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        layer_outputs = self.layer(
            x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[index], 
            output_attentions=output_attentions
        )
        hidden_state = layer_outputs[-1]

        if output_attentions:
            assert len(layer_outputs) == 2
            attentions = layer_outputs[0]
            all_attentions = all_attentions + (attentions,)
        else:
            assert len(layer_outputs) == 1

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # shrink all_hidden_states
        all_hidden_states = all_hidden_states[-1:]

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

class TransforemrIntermediateBlock(torch.nn.Module):
    def __init__(self, block) -> None:
        super().__init__()
        self.layer = block
    
    def forward(self, inputs):
        hidden_state, attn_mask, head_mask, \
        output_attentions, output_hidden_states, return_dict, \
        all_hidden_states, all_attentions, index = inputs

        if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        layer_outputs = self.layer(
            x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[index], 
            output_attentions=output_attentions
        )
        hidden_state = layer_outputs[-1]

        if output_attentions:
            assert len(layer_outputs) == 2
            attentions = layer_outputs[0]
            all_attentions = all_attentions + (attentions,)
        else:
            assert len(layer_outputs) == 1


        # for hidden states, only keep the last encoder output
        if all_hidden_states is not None:
            all_hidden_states = all_hidden_states[-1:]

        next_block_input = [hidden_state, attn_mask, head_mask, 
                            output_attentions, output_hidden_states, return_dict,
                            all_hidden_states, all_attentions, index + 1]
        
        return next_block_input

class TransformerFirstBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.layer = block

    def forward(self, inputs):
        x, attn_mask, head_mask, \
        output_attentions, output_hidden_states, return_dict = inputs

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        index = 0 

        hidden_state = x

        if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        layer_outputs = self.layer(
            x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[index], 
            output_attentions=output_attentions
        )
        hidden_state = layer_outputs[-1]

        if output_attentions:
            assert len(layer_outputs) == 2
            attentions = layer_outputs[0]
            all_attentions = all_attentions + (attentions,)
        else:
            assert len(layer_outputs) == 1
        
        next_block_input = [hidden_state, attn_mask, head_mask, 
                            output_attentions, output_hidden_states, return_dict,
                            all_hidden_states, all_attentions, index + 1]
        
        return next_block_input


class DistilBertModelInitWrapper(torch.nn.Module):
    def __init__(self, embedding, config):
        super().__init__()
        self.model = DistilBertModelInit(embedding, config)
    
    def forward(self, input: Dict):
        return self.model(**input)
    
class DistilBertModelInit(torch.nn.Module):
    def __init__(self, embedding, config):
        super().__init__()
        self.embeddings = embedding
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        
        transformer_input = [inputs_embeds, attention_mask, head_mask, 
                                output_attentions, output_hidden_states, return_dict]
        
        return transformer_input
    
    def get_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False):
        
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
def main():
    #configuration = BertConfig()
    student_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    print(student_model)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print(inputs)
    sequential_layers = ditilbert_to_sequential(student_model, inputs)
    
if __name__ == '__main__':
    main()
