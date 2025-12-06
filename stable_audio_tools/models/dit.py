import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .blocks import FourierFeatures
from .transformer import ContinuousTransformer

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        input_add_dims={},
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        causal=False,
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            if timestep_embed_dim is None:
                timestep_embed_dim = embed_dim
                print(f"Using default timestep_embed_dim={timestep_embed_dim} for input_concat timestep conditioning")
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        # if timestep_cond_type == "global" and global_cond_type == "adaLN":
        #     # zero init the second layer of the timestep embedding MLP to stabilize training
        #     nn.init.zeros_(self.to_timestep_embed[2].weight)
        #     nn.init.zeros_(self.to_timestep_embed[2].bias)
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if input_add_dims is not None:
            # Input add conditioning, module dict
            self.to_input_add_embed = nn.ModuleDict()
            for id, dim in input_add_dims.items():
                self.to_input_add_embed[id] = nn.Linear(dim, embed_dim, bias=False)
            self.input_add_cond_cache = None

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type
        self.causal = causal

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                causal=causal,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)


    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        input_add_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        exit_layer_ix=None,
        use_input_add_cond_cache=False,
        **kwargs):

        use_cross_kv_cache = cross_attn_cond is not None and all([self.transformer.layers[i].cross_attn.kv_cache is not None and self.transformer.layers[i].cross_attn.kv_cache.is_init for i in range(len(self.transformer.layers))])
        # use_cross_kv_cache = False
        
        if cross_attn_cond is not None and not use_cross_kv_cache:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        add_emb = None
        if input_add_cond is not None:
            # Project the input add conditioning to the io_channels dimension
            if use_input_add_cond_cache and self.input_add_cond_cache is not None:
                add_emb = self.input_add_cond_cache
                input_pos = kwargs.get('input_pos', None)
                add_emb = add_emb[:, input_pos]
            else:
                for k, v in input_add_cond.items():
                    if k in self.to_input_add_embed:
                        emb = self.to_input_add_embed[k](v.transpose(1, 2))  # (b, t, c) -> (b, c, t)
                        # if emb.shape[1] == x.shape[1]:
                        #     x += emb.transpose(1, 2)
                        # else:
                        if add_emb is None:
                            add_emb = emb
                        else:
                            add_emb += emb
                    else:
                        print(f"Unknown input_add_cond key: {k}")
                if use_input_add_cond_cache:
                    self.input_add_cond_cache = add_emb.detach().clone() 
                    input_pos = kwargs.get('input_pos', None)
                    add_emb = add_emb[:, input_pos]
                

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t.unsqueeze(-1))) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(-1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "continuous_transformer":
            # Masks not currently implemented for continuous transformer
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, return_info=return_info, exit_layer_ix=exit_layer_ix, input_add_emb=add_emb, **extra_args, **kwargs)

            if return_info:
                output, info = output

            # Avoid postprocessing on early exit
            if exit_layer_ix is not None:
                if return_info:
                    return output, info
                else:
                    return output

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def apg_project(self, v0, v1):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def forward(
        self, 
        x, 
        t, 
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        input_add_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_norm_threshold=0.0,
        cfg_interval = (0, 1),
        scale_phi=0.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        # assert causal == False, "Causal mode is not supported for DiffusionTransformer"
        # if input_add_cond is None:
        #     print("WARNING: input_add_cond is None, this may lead to unexpected behavior if the model expects input_add_cond to be present.")

        model_dtype = next(self.parameters()).dtype
        # if self.causal:
        #     print(f"time step {t} with shape {x.shape}")
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        if input_add_cond is not None:
            # Convert input_add_cond to the model dtype
            for k, v in input_add_cond.items():
                input_add_cond[k] = v.to(model_dtype)

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                input_add_cond=input_add_cond,
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:

            # with cfg_dropout_prob, dropout everything together, and also with cfg_dropout_prob drop each individually
            global_mask = torch.bernoulli(torch.full((1, 1, 1), cfg_dropout_prob, device=x.device)).to(torch.bool)
        

            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

                # dropout from global_mask, first reshape to match cross_attn_cond
                cross_attn_global_mask = global_mask.expand(cross_attn_cond.shape[0], -1, -1)
                cross_attn_cond = torch.where(cross_attn_global_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

                # dropout from global_mask, first reshape to match prepend_cond
                prepend_global_mask = global_mask.expand(prepend_cond.shape[0], -1, -1)
                prepend_cond = torch.where(prepend_global_mask, null_embed, prepend_cond)

            if input_add_cond is not None:
                for k, v in input_add_cond.items():
                    null_embed = torch.zeros_like(v, device=v.device)
                    dropout_mask = torch.bernoulli(torch.full((v.shape[0], 1, 1), cfg_dropout_prob, device=v.device)).to(torch.bool)
                    input_add_cond[k] = torch.where(dropout_mask, null_embed, v)

                    # dropout from global_mask, first reshape to match input_add_cond
                    input_add_global_mask = global_mask.expand(v.shape[0], -1, -1)
                    input_add_cond[k] = torch.where(input_add_global_mask, null_embed, input_add_cond[k])

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        if (((type(cfg_scale) == float or type(cfg_scale) == int) and cfg_scale != 1.0) or (type(cfg_scale) == list and any([z != 1.0 for z in cfg_scale])) and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1])):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension

            if type(cfg_scale) == float or type(cfg_scale) == int:
                batch_inputs = torch.cat([x, x], dim=0)
                batch_timestep = torch.cat([t, t], dim=0)

                if global_embed is not None:
                    batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
                else:
                    batch_global_cond = None

                if input_concat_cond is not None:
                    batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
                else:
                    batch_input_concat_cond = None

                batch_cond = None
                batch_cond_masks = None
                
                # Handle CFG for cross-attention conditioning
                if cross_attn_cond is not None:

                    null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                    # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                    if negative_cross_attn_cond is not None:

                        # If there's a negative cross-attention mask, set the masked tokens to the null embed
                        if negative_cross_attn_mask is not None:
                            negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                            negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                        
                        batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                    else:
                        batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                    if cross_attn_cond_mask is not None:
                        batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
                
                batch_prepend_cond = None
                batch_prepend_cond_mask = None

                if prepend_cond is not None:

                    null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                    batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                            
                    if prepend_cond_mask is not None:
                        batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
            

                if mask is not None:
                    batch_masks = torch.cat([mask, mask], dim=0)
                else:
                    batch_masks = None

                # now handle input_add_cond
                batch_input_add_cond = {}
                if input_add_cond is not None:
                    for k, v in input_add_cond.items():
                        # basically in this setup the input_add_cond is never dropped out, so the "unconditional" branch in this case is the "non-text-conditional", rather than full unconditional
                        # this is equivalent to eq. (3) from https://arxiv.org/pdf/2211.09800 when S_I = 1
                        # null_embed = torch.zeros_like(v, device=v.device)
                        batch_input_add_cond[k] = torch.cat([v, v], dim=0)
                
                batch_output = self._forward(
                    batch_inputs, 
                    batch_timestep, 
                    cross_attn_cond=batch_cond, 
                    input_add_cond=batch_input_add_cond,
                    cross_attn_cond_mask=batch_cond_masks, 
                    mask = batch_masks, 
                    input_concat_cond=batch_input_concat_cond, 
                    global_embed = batch_global_cond,
                    prepend_cond = batch_prepend_cond,
                    prepend_cond_mask = batch_prepend_cond_mask,
                    return_info = return_info,
                    **kwargs)

                if return_info:
                    batch_output, info = batch_output

                cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
                
                if self.diffusion_objective == "v":
                    cond_denoised = x * alpha[:, None, None] - cond_output * sigma[:, None, None]
                    uncond_denoised = x * alpha[:, None, None] - uncond_output * sigma[:, None, None]

                elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                    cond_denoised = x - cond_output * sigma[:, None, None]
                    uncond_denoised = x - uncond_output * sigma[:, None, None]

                diff = cond_denoised - uncond_denoised
                
                if cfg_norm_threshold > 0:
                    diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
                    scale_factor = torch.minimum(torch.ones_like(diff), cfg_norm_threshold / diff_norm)
                    diff *= scale_factor

                diff_parallel, diff_orthogonal = self.apg_project(diff, cond_denoised)

                cfg_diff = diff_orthogonal

                cfg_denoised = cond_denoised + (cfg_scale - 1) * cfg_diff
                        
                if self.diffusion_objective == "v":
                    output = (x * alpha[:, None, None] - cfg_denoised) / sigma[:, None, None]
                elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                    output = (x - cfg_denoised) / sigma[:, None, None]

                # CFG Rescale
                if scale_phi != 0.0:
                    cond_out_std = cond_output.std(dim=1, keepdim=True)
                    out_cfg_std = output.std(dim=1, keepdim=True)
                    output = scale_phi * (output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * output
            else:
                # THIS IS FOR MULTICFG
                batch_inputs = torch.cat([x, x, x], dim=0)
                batch_timestep = torch.cat([t, t, t], dim=0)

                if global_embed is not None:
                    batch_global_cond = torch.cat([global_embed, global_embed, global_embed], dim=0)
                else:
                    batch_global_cond = None

                if input_concat_cond is not None:
                    batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond, input_concat_cond], dim=0)
                else:
                    batch_input_concat_cond = None

                batch_cond = None
                batch_cond_masks = None

                # Handle CFG for cross-attention conditioning
                if cross_attn_cond is not None:

                    null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                    # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                    if negative_cross_attn_cond is not None:

                        # If there's a negative cross-attention mask, set the masked tokens to the null embed
                        if negative_cross_attn_mask is not None:
                            negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                            negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                        
                        batch_cond = torch.cat([cross_attn_cond, cross_attn_cond, negative_cross_attn_cond], dim=0)

                    else:
                        batch_cond = torch.cat([cross_attn_cond, null_embed, null_embed], dim=0)

                    if cross_attn_cond_mask is not None:
                        batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask, cross_attn_cond_mask], dim=0)

                batch_prepend_cond = None
                batch_prepend_cond_mask = None

                if prepend_cond is not None:

                    null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                    batch_prepend_cond = torch.cat([prepend_cond, null_embed, null_embed], dim=0)
                            
                    if prepend_cond_mask is not None:
                        batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask, prepend_cond_mask], dim=0)
            

                if mask is not None:
                    batch_masks = torch.cat([mask, mask, mask], dim=0)
                else:
                    batch_masks = None


                # now handle input_add_cond
                batch_input_add_cond = {}
                for k, v in input_add_cond.items():
                    null_embed = torch.zeros_like(v, device=v.device)
                    batch_input_add_cond[k] = torch.cat([v, v, null_embed], dim=0)

                batch_output = self._forward(
                    batch_inputs, 
                    batch_timestep, 
                    cross_attn_cond=batch_cond, 
                    input_add_cond=batch_input_add_cond,
                    cross_attn_cond_mask=batch_cond_masks, 
                    mask = batch_masks, 
                    input_concat_cond=batch_input_concat_cond, 
                    global_embed = batch_global_cond,
                    prepend_cond = batch_prepend_cond,
                    prepend_cond_mask = batch_prepend_cond_mask,
                    return_info = return_info,
                    **kwargs)

                if return_info:
                    batch_output, info = batch_output

                cond_output, cond_add_no_others_output, uncond_output = torch.chunk(batch_output, 3, dim=0)
                if type(cfg_scale) == float:
                    cfg_output = uncond_output + cfg_scale * (cond_add_no_others_output - uncond_output) + cfg_scale * (cond_output - cond_add_no_others_output)
                elif type(cfg_scale) == list:
                    cfg_output = uncond_output + (cond_add_no_others_output - uncond_output) * cfg_scale[0] + (cond_output - cond_add_no_others_output) * cfg_scale[1]

                # CFG Rescale
                if scale_phi != 0.0:
                    cond_out_std = cond_output.std(dim=1, keepdim=True)
                    out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                    output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
                else:
                    output = cfg_output
                
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                input_add_cond=input_add_cond,
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )