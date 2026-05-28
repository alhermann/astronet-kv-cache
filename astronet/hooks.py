"""
Hook-based integration of AstroNet with frozen HuggingFace LLMs.

Uses PyTorch forward hooks to inject astrocytic feedback into the
transformer's attention computation without modifying model code.

Strategy:
  Pre-hook on target layer's self_attn module adds a bias to hidden states
  BEFORE they are projected into K, Q, V. Since K = W_k @ (h + bias),
  this is equivalent to K' = K_original + W_k @ bias — effectively
  biasing what the attention layer treats as "key" information.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from astronet.calcium import AstroStateV0


class AstroWrappedModel(nn.Module):
    """
    Wraps a frozen LLM with an AstroState module.
    Only AstroState parameters are trainable.

    The wrapper:
    1. Freezes all base model parameters
    2. Installs forward hooks for astrocytic modulation
    3. After each forward pass, updates the astrocytic state
    4. Exposes only AstroNet parameters for optimization

    Args:
        base_model: A HuggingFace causal LM (e.g. LlamaForCausalLM)
        astro_state: An AstroStateV0 instance
    """

    def __init__(self, base_model: nn.Module, astro_state: AstroStateV0):
        super().__init__()
        self.base_model = base_model
        self.astro = astro_state
        self._hooks = []

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Install hooks
        self._install_hooks()

    def _get_target_attention_module(self) -> nn.Module:
        """
        Find the attention module at the target layer.
        Supports Llama, Mistral, Qwen model structures.
        """
        target_layer = self.astro.target_layer

        # Try common HuggingFace model structures
        if hasattr(self.base_model, 'model'):
            inner = self.base_model.model
        elif hasattr(self.base_model, 'transformer'):
            inner = self.base_model.transformer
        else:
            raise ValueError(
                f"Unknown model structure: {type(self.base_model)}. "
                "Expected .model or .transformer attribute."
            )

        if hasattr(inner, 'layers'):
            layers = inner.layers
        elif hasattr(inner, 'h'):
            layers = inner.h
        else:
            raise ValueError(
                f"Cannot find layers in {type(inner)}. "
                "Expected .layers or .h attribute."
            )

        if target_layer >= len(layers):
            raise ValueError(
                f"target_layer={target_layer} exceeds model depth ({len(layers)} layers)"
            )

        layer = layers[target_layer]

        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        elif hasattr(layer, 'attn'):
            return layer.attn
        else:
            raise ValueError(
                f"Cannot find attention module in layer {target_layer}. "
                "Expected .self_attn or .attn attribute."
            )

    def _install_hooks(self) -> None:
        """Install hook for astrocytic modulation.

        Injects feedback as a bias on the input to lm_head (the last hidden
        states), which directly biases output logits via the frozen lm_head.
        This gives the shortest gradient path from feedback to loss.
        """
        # Find the lm_head module
        if hasattr(self.base_model, 'lm_head'):
            lm_head = self.base_model.lm_head
        else:
            raise ValueError(
                f"Cannot find lm_head in {type(self.base_model)}."
            )

        def pre_lm_head_hook(module, args):
            """
            Pre-hook on lm_head: add astrocytic bias to the final hidden states
            before they're projected into logits.

            The feedback is added to ALL token positions. The frozen lm_head
            then converts (h + feedback) into logits, effectively adding a
            bias in logit space proportional to lm_head.weight @ feedback.
            """
            feedback_fp32 = self.astro.get_feedback()  # (1, hidden_dim)

            if len(args) > 0:
                hidden_states = args[0]
                feedback = feedback_fp32.to(dtype=hidden_states.dtype)
                modulated = hidden_states + feedback
                return (modulated,) + args[1:]
            return args

        handle = lm_head.register_forward_pre_hook(pre_lm_head_hook)
        self._hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        update_astro: bool = True,
        keep_grad: bool = False,
    ) -> dict:
        """
        Forward pass through the wrapped model.

        1. The pre-hook injects astrocytic bias into the target layer
        2. The base model processes the input normally
        3. After the forward pass, the astrocytic state is updated
           from the target layer's hidden states

        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len) attention mask
            labels: (batch, seq_len) target token ids for loss computation
            update_astro: Whether to update astrocytic state after this pass
            keep_grad: Whether to keep gradients on the astro state update

        Returns:
            Model outputs (with loss if labels provided)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        if update_astro:
            # Extract hidden states from the target layer
            target_hidden = outputs.hidden_states[self.astro.target_layer]
            sensed = self.astro.sense(target_hidden)
            self.astro.update_state(sensed, keep_grad=keep_grad)

        return outputs

    def trainable_parameters(self):
        """Yield only the trainable (AstroNet) parameters."""
        return self.astro.parameters()

    def trainable_parameter_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    def freeze_check(self) -> dict:
        """Verify that only AstroNet parameters are trainable."""
        base_trainable = sum(
            1 for p in self.base_model.parameters() if p.requires_grad
        )
        astro_trainable = sum(
            1 for p in self.astro.parameters() if p.requires_grad
        )
        return {
            'base_model_trainable_params': base_trainable,
            'astronet_trainable_params': astro_trainable,
            'freeze_correct': base_trainable == 0,
        }


def install_astro_hooks(
    model: nn.Module,
    astro_state: AstroStateV0,
) -> 'AstroWrappedModel':
    """
    Convenience function: wrap a model with astrocytic modulation.

    Args:
        model: A HuggingFace causal LM
        astro_state: An AstroStateV0 instance

    Returns:
        AstroWrappedModel with hooks installed
    """
    return AstroWrappedModel(model, astro_state)
