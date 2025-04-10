"""
This module provides the fused linear cross entropy (LCE) forward function for the Llama-4
conditional generation model when using Liger kernels. This new forward function (exposed
as `llama_4_lce_forward`) is adapted to the Llama-4 architecture (i.e. as implemented in
`src/transformers/models/llama4/modeling_llama4.py`) and is designed to integrate Liger’s fused
linear cross entropy loss implementation into the model’s forward pass.
"""

from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

# Import the output type for Llama-4; adjust the import path as needed
from transformers.models.llama4.modeling_llama4 import Llama4CausalLMOutputWithPast

# Import the fused cross entropy loss implementation provided by Liger.
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

# We also need to import any other utilities (if available) that are used by similar fused loss functions.
# For example, if helper functions for shifting tokens or masks are defined elsewhere, import them here.
# In this example, we inline the shift and flatten logic.


def lce_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, Llama4CausalLMOutputWithPast]:
    """
    Llama-4 adapted forward function using Liger’s fused linear cross entropy.
    
    This function is intended to replace the standard forward method for Llama4ForConditionalGeneration
    when fused cross entropy loss is desired. It computes the logits via the model’s usual forward pass,
    then (when training and if `labels` is provided) computes the loss in a memory–efficient way using
    LigerFusedLinearCrossEntropyLoss.

    Args:
        input_ids (torch.LongTensor, optional): Token IDs for the sequence.
        attention_mask (torch.Tensor, optional): Attention mask for the inputs.
        position_ids (torch.LongTensor, optional): Position IDs for positional embeddings.
        past_key_values (List[torch.FloatTensor], optional): Cached past key/value states.
        inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings.
        labels (torch.LongTensor, optional): Labels for language modeling.
        use_cache (bool, optional): Whether to use caching for fast autoregressive generation.
        output_attentions (bool, optional): Whether to return attention weights.
        output_hidden_states (bool, optional): Whether to return all hidden states.
        return_dict (bool, optional): Whether to return a ModelOutput object.
        cache_position (torch.LongTensor, optional): Cache positions for updating key/value caches.
        **kwargs: Additional keyword arguments to be passed to the underlying forward call.

    Returns:
        Either a tuple or a Llama4CausalLMOutputWithPast, containing (loss, logits, past_key_values, hidden_states, attentions)
        if `return_dict` is True, otherwise a tuple.
    """
    # Ensure that exactly one of input_ids and inputs_embeds is provided.
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

    # Obtain the input embeddings if they were not directly provided.
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # If cache_position is not provided, derive it from past_key_values if available.
    if cache_position is None:
        if past_key_values is not None and len(past_key_values) > 0:
            # Assume that the first tensor in past_key_values has shape [batch_size, ..., seq_len, ...]
            past_length = past_key_values[0][0].shape[-2]
        else:
            past_length = 0
        cache_position = torch.arange(
            past_length, past_length + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # If position_ids is not provided, derive from cache_position.
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Call the underlying forward function of Llama4ForConditionalGeneration
    # Note: The underlying model is assumed to be available as `self.model` (or similar) and
    # returns an output structure that includes logits, past_key_values, etc.
    output = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    if return_dict:
        logits = output.logits
    else:
        logits = output[0]

    loss = None
    # If we are training and labels are provided, compute the fused cross entropy loss.
    if self.training and labels is not None:
        # Shift logits and labels so that each token predicts the next token.
        # If an attention mask is provided, use it to select only the attended positions.
        if attention_mask is not None:
            # Compute a shift mask for tokens that are not masked.
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
            # Extract logits for all but the last token.
            shift_logits = logits[..., :-1, :]
            # Extract corresponding labels (skip the first token).
            shift_labels = labels[..., 1:]
            # Now apply the attention mask to filter positions.
            shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        # Flatten the logits and labels.
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1).to(shift_logits.device)

        # Use Liger’s fused linear cross entropy loss.
        loss_fct = LigerFusedLinearCrossEntropyLoss()
        loss = loss_fct(self.lm_head.weight, shift_logits, shift_labels)

    # Return the outputs in the same format as the underlying model.
    if not return_dict:
        # Assume underlying output is a tuple: first element logits and then past_key_values, etc.
        output = (logits,) + output[1:]
        return (loss,) + output if loss is not None else output

    return Llama4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=output.past_key_values,
        hidden_states=output.hidden_states,
        attentions=output.attentions,
    )


def lce_vision_forward(
    self,
    pixel_values: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, Llama4CausalLMOutputWithPast]:
    """
    Llama-4 forward function re-integrating the pixel input processing component.
    
    If `pixel_values` is provided (as is typical for vision-enabled variants),
    the image inputs are pre-processed via patch embeddings, class token injection,
    and positional embeddings.
    Otherwise, the function will derive the input embeddings from `input_ids`.
    """
    # Decide whether to execute vision- or text-based processing.
    if pixel_values is not None:
        # Process the image input using the vision components.
        # Expected shape: [batch_size * num_tiles, channels, height, width]
        batch_size_times_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1

        # Obtain patch embeddings from the pixel values.
        hidden_state = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_state.shape

        # Inject a class token by reshaping and concatenation.
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media * num_chunks, num_patches, hidden_dim
        )
        class_embedding = self.class_embedding.expand(hidden_state.shape[0], 1, hidden_state.shape[-1])
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Apply positional embeddings.
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media, num_chunks, num_patches, hidden_dim
        )
        positional_embedding = self.positional_embedding_vlm.to(dtype=hidden_state.dtype, device=hidden_state.device)
        hidden_state = hidden_state + positional_embedding

        # Normalize the embeddings.
        hidden_state = self.layernorm_pre(hidden_state)
        hidden_state = hidden_state.view(batch_size_times_num_tiles, -1, hidden_dim)

        # Compute rotary embeddings (or any frequency-based conditioning) for vision.
        freqs_ci = self.rotary_embedding(pixel_values)
    else:
        # Ensure exactly one of input_ids or inputs_embeds is provided.
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        hidden_state = inputs_embeds

        # If no vision inputs are provided, you can set dummy freqs or process position IDs normally.
        freqs_ci = None

    # If cache_position is not provided, derive it.
    if cache_position is None:
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = past_key_values[0][0].shape[-2]
        else:
            past_length = 0
        # Assume inputs_embeds (or hidden_state from images) has shape [batch, seq_length, ...]
        cache_position = torch.arange(
            past_length, past_length + hidden_state.shape[1],
            device=hidden_state.device
        )

    # If no explicit position_ids are provided, use the cache positions.
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Continue by calling the underlying Llama4 model.
    output = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=hidden_state,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        freqs_ci=freqs_ci,  # Pass vision-specific frequencies if computed.
        **kwargs,
    )

    # Retrieve logits from the output.
    logits = output.logits if return_dict else output[0]

    loss = None
    if self.training and labels is not None:
        if attention_mask is not None:
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1):].to(logits.device)
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss_fct = LigerFusedLinearCrossEntropyLoss()
        loss = loss_fct(self.lm_head.weight, shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + output[1:]
        return (loss,) + output if loss is not None else output

    return Llama4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=output.past_key_values,
        hidden_states=output.hidden_states,
        attentions=output.attentions,
    )