
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge
from tomesd.merge import create_windows, calculate_similarity, merge_windows
from .utils import isinstance_str, init_generator

# def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
#     """
#     Compute merge functions for the current layer
#     Args:
#         x: input tensor
#         tome_info: dictionary containing merge parameters and size information
#     """
#     original_h, original_w = tome_info["size"]
#     original_tokens = original_h * original_w
#     downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

#     args = tome_info["args"]

#     if downsample <= args["max_downsample"]:
#         w = int(math.ceil(original_w / downsample))
#         h = int(math.ceil(original_h / downsample))
        
#         # Re-init the generator if needed
#         if args["generator"] is None:
#             args["generator"] = init_generator(x.device)
#         elif args["generator"].device != x.device:
#             args["generator"] = init_generator(x.device, fallback=args["generator"])
        
#         # Multi-scale token merging
#         m, u = merge.window_based_soft_matching(  # 수정된 함수 호출
#             input_tensor=x,
#             w=w, 
#             h=h,
#             window_size=args["window_sizes"],
#             r=int(args["ratio"] * x.shape[1]),
#             # no_rand=not args["use_rand"],
#             # generator=args["generator"]
#         )
#     else:
#         m, u = (merge.do_nothing, merge.do_nothing)

#     # Apply merging to different parts of the network based on args
#     m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
#     m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
#     m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

#     return m_a, m_c, m_m, u_a, u_c, u_m
def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    # print("downsample: ", downsample)
    # print("args[max_downsample], ",args["max_downsample"])
    if downsample <= args["max_downsample"]:

        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        result, unmerge_fn  = merge.window_based_soft_matching(
            input_tensor=x,
            w=w, 
            h=h,
            r=r,
            mode="mean"
        )
        m = result
        u = unmerge_fn
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    # m_a, u_a = (m, u) if args["merge_attn"] else (lambda x: x, lambda x: x)
    # m_c, u_c = (m, u) if args["merge_crossattn"] else (lambda x: x, lambda x: x)
    # m_m, u_m = (m, u) if args["merge_mlp"] else (lambda x: x, lambda x: x)
    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


# def apply_patch(
#         model: torch.nn.Module,
#         ratio: float = 0.5,
#         window_sizes: int = 4,  # 사용할 window size들
#         merge_ratios: float = 0.3,  # 각 window size별 merge ratio
#         use_rand: bool = True,
#         max_downsample: int = 1,
#         merge_attn: bool = True,
#         merge_crossattn: bool = False,
#         merge_mlp: bool = False):
#     """
#     Applies hierarchical window-based token merging to a model
    
#     Args:
#         model: model to patch
#         ratio: overall token reduction ratio
#         window_sizes: list of window sizes to use (e.g., [8, 16])
#         merge_ratios: list of merge ratios for each window size
#         use_rand: whether to use randomization
#         max_downsample: maximum downsampling factor
#         merge_attn: whether to merge in self-attention
#         merge_crossattn: whether to merge in cross-attention
#         merge_mlp: whether to merge in MLP
#     """
#     remove_patch(model)
    
#     is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
#     if not is_diffusers:
#         if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
#             raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model")
#         diffusion_model = model.model.diffusion_model
#     else:
#         diffusion_model = model.unet if hasattr(model, "unet") else model
    
#     # Store parameters in model's tome_info
#     diffusion_model._tome_info = {
#         "size": None,
#         "hooks": [],
#         "args": {
#             "ratio": ratio,
#             "window_sizes": window_sizes,
#             "merge_ratios": merge_ratios,
#             "use_rand": use_rand,
#             "max_downsample": max_downsample,
#             "generator": None,
#             "merge_attn": merge_attn,
#             "merge_crossattn": merge_crossattn,
#             "merge_mlp": merge_mlp
#         }
#     }
    
#     hook_tome_model(diffusion_model)
    
#     # Patch transformer blocks
#     for _, module in diffusion_model.named_modules():
#         if isinstance_str(module, "BasicTransformerBlock"):
#             make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
#             module.__class__ = make_tome_block_fn(module.__class__)
#             module._tome_info = diffusion_model._tome_info
            
#             if not hasattr(module, "disable_self_attn") and not is_diffusers:
#                 module.disable_self_attn = False
                
#             if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
#                 module.use_ada_layer_norm = False
#                 module.use_ada_layer_norm_zero = False
    
#     return model
def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    # window_size: int = 4,  # 윈도우 크기

    use_rand: bool = True,
    max_downsample: int = 1,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False
):
    """
    Patches a model with hierarchical window-based token merging.

    Args:
        model: Model to patch.
        ratio: Overall token reduction ratio.
        window_size: Size of the window for merging.
        merge_ratios: Merge ratio for windows.
        use_rand: Whether to use randomness.
        max_downsample: Maximum downsampling factor.
        merge_attn: Whether to merge tokens in self-attention layers.
        merge_crossattn: Whether to merge tokens in cross-attention layers.
        merge_mlp: Whether to merge tokens in MLP layers.
    """
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model.")
        diffusion_model = model.model.diffusion_model
    else:
        diffusion_model = model.unet if hasattr(model, "unet") else model

    # Store parameters in the model's ToMe info
    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            # "window_sizes": window_size,
            "use_rand": use_rand,
            "max_downsample": max_downsample,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }

    hook_tome_model(diffusion_model)

    # Patch transformer blocks
    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model


# The rest of the code (make_tome_block, make_diffusers_tome_block, hook_tome_model, remove_patch)
# remains the same as 
def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
