import pytorch_lightning as pl
import random
import torch
import typing as tp

from ema_pytorch import EMA
from torch.nn import functional as F

from ..inference.sampling import truncated_logistic_normal_rescaled
from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..models.inpainting import random_inpaint_mask
from .utils import create_optimizer_from_config, create_scheduler_from_config

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

def euler_step(x_t, v_t, t, s):
    # manage s and t ndimensions
    if s.ndim == 1:
        s = s[:, None, None]
    elif s.ndim == 2:
        s = s[:, None]
    if t.ndim == 1:
        t = t[:, None, None]
    elif t.ndim == 2:
        t = t[:, None]

    return x_t + (s - t) * v_t

# def euler_step(x_t, v_t, t, s):
#     return x_t + (s - t)[:, None, None] * v_t

@torch.no_grad()
def sample_flow_dpmpp_w_intermediates(model, x, sigmas=None, steps=None, callback=None, dist_shift=None, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models. Return output at each step."""

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        # Create the noise schedule
        t = torch.linspace(1, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])
    
    else:
        t = sigmas

    old_denoised = None

    log_snr = lambda t: ((1-t) / t).log()
    inters_x = []
    inters_t = []

    for i in range(len(t) - 1):
        inters_x.append(x)
        inters_t.append(t[i])
        denoised = x - t[i] * model(x, t[i] * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})
        t_curr, t_next = t[i], t[i + 1]
        alpha_t = 1-t_next
        h = log_snr(t_next) - log_snr(t_curr)
        if old_denoised is None or t_next == 0:
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised_d
        old_denoised = denoised
    target = x.detach()
    inters_x = torch.stack(inters_x).detach() # steps x B x C x T
    inters_t = torch.stack(inters_t).unsqueeze(-1).detach() # steps x 1

    return {'target': target, 'x': inters_x, 't': inters_t}

class ARCTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for ARC post-training on a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            discriminator: ConditionedDiffusionModelWrapper,
            arc_config: dict,
            optimizer_configs: dict,
            teacher_model: ConditionedDiffusionModelWrapper = None,
            use_ema: bool = True,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.0,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal"] = "uniform",
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            clip_grad_norm: float = 0.0,
            trim_config = None,
            inpainting_config = None,
            diffusion_forcing: bool = False,
    ):
        super().__init__()

        self.automatic_optimization = False


        self.diffusion = model

        self.teacher_model = teacher_model

        if self.teacher_model is not None:
            self.teacher_model.eval().requires_grad_(False)

        self.discriminator = discriminator

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.ode_warmup_config = arc_config.get('ode_warmup', None)
        self.dmd_weight = arc_config.get("dmd_weight", 1.0)
        self.fake_dsm_weight = arc_config.get("fake_dsm_weight", 1.0)

        if self.ode_warmup_config is not None:

            self.ode_warmup_steps = self.ode_warmup_config.get('warmup_steps', 0)
            self.ode_refresh_rate = self.ode_warmup_config.get('refresh_rate', 1)
            self.ode_n_sampling_steps = self.ode_warmup_config.get('sampling_steps', 20)
            self.ode_warmup_cfg = self.ode_warmup_config.get('cfg', 4.0)
            self.ode_step_divisor = self.ode_warmup_config.get('step_divisor', self.ode_n_sampling_steps)
        else:
            self.ode_warmup_steps = 0

        sampling_config = arc_config.get('sampling', None)

        self.diff_states = []

        self.noise_dist_config = arc_config.get('noise_dist', {})
        self.gen_noise_dist = self.build_noise_dist('generator')
        self.dis_noise_dist = self.build_noise_dist('discriminator')
        self.dmd_noise_dist = self.build_noise_dist('dmd')
        self.fake_dsm_noise_dist = self.build_noise_dist('fake_dsm')


        self.discriminator_config = arc_config.get('discriminator', {})
        self.diffusion_forcing = diffusion_forcing

        if self.diffusion_forcing:
            # if diffusion forcing, timestep conditioning NEEDS to be in input_concat mode
            pass
            # assert self.diffusion.model.timestep_cond_type == 'input_concat', "Diffusion forcing requires timestep conditioning to be in input_concat mode"

        self.discriminator_dit_layer = self.discriminator_config.get('dit_hidden_layer', None)

        self.do_contrastive_disc = self.discriminator_config.get('contrastive', False)
        
        self.include_grad_penalties = self.discriminator_config.get('include_grad_penalties', False)

        assert self.discriminator_dit_layer is not None, "Must specify discriminator dit_hidden_layer in ARC config"
        # delete all layers after the specified layer in the discriminator, as those will never be used
        # self.discriminator.model.model.transformer.layers = self.discriminator.model.model.transformer.layers[:self.discriminator_dit_layer+1]
        # torch.cuda.empty_cache()

        discriminator_type = self.discriminator_config.get('type', 'convnext')
        discriminator_model_config = self.discriminator_config.get('config', {})

        if discriminator_type == 'convnext':
            from ..models.arc import ConvNeXtDiscriminator
            self.discriminator_head = ConvNeXtDiscriminator(in_channels = self.discriminator.model.model.transformer.dim, latent_dim=1, **discriminator_model_config)
        elif discriminator_type == 'conv':
            from ..models.arc import ConvDiscriminator
            self.discriminator_head = ConvDiscriminator(channels = self.discriminator.model.model.transformer.dim, **discriminator_model_config)

        self.gen_gan_weight = self.discriminator_config.get('weights', {}).get('generator', 1.0)
        self.dis_gan_weight = self.discriminator_config.get('weights', {}).get('discriminator', 1.0)

        if self.do_contrastive_disc:
            self.contrastive_loss_weight = self.discriminator_config.get('weights', {}).get('contrastive', 1.0)

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.diffusion_objective = model.diffusion_objective

        assert optimizer_configs is not None, "Must specify optimizer_configs in training config"

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        self.model_last_layer = self.diffusion.model.model.transformer.project_out.weight

        self.clip_grad_norm = clip_grad_norm

        self.trim_config = trim_config

        if self.trim_config is not None:
            self.trim_prob = self.trim_config.get("trim_prob", 0.0)
            self.trim_type = self.trim_config.get("type", "random_item")

        self.inpainting_config = inpainting_config

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})
            self.ode_inpaint_mask = None
            self.ode_inpaint_masked_input = None

        # Validation

        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        disc_opt_config = self.optimizer_configs['discriminator']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())
        opt_disc = create_optimizer_from_config(disc_opt_config['optimizer'], list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()))
        if "scheduler" in diffusion_opt_config and "scheduler" in disc_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            sched_disc = create_scheduler_from_config(disc_opt_config['scheduler'], opt_disc)
            sched_disc_config = {
                "scheduler": sched_disc,
                "interval": "step"
            }
            return [opt_diff, opt_disc], [sched_diff_config, sched_disc_config]

        return [opt_diff, opt_disc]

    def ode_warmup_step(self, diffusion_input, metadata, padding_masks):
        refresh_diff_states = self.global_step % self.ode_refresh_rate == 0

        if refresh_diff_states:
            start_noise = torch.randn_like(diffusion_input)
            teacher_conditioning = self.teacher_model.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                # Create a mask of random length for a random slice of the input
                inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

                teacher_conditioning['inpaint_mask'] = [inpaint_mask]
                teacher_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                self.ode_inpaint_mask = inpaint_mask
                self.ode_inpaint_masked_input = inpaint_masked_input

            logsnr = torch.linspace(-6, 2, self.ode_n_sampling_steps + 1)
            t = torch.sigmoid(-logsnr)
            t[0] = 1
            t[-1] = 0

            self.ode_metadata = metadata
            self.diff_states = sample_flow_dpmpp_w_intermediates(self.teacher_model, start_noise, sigmas=t, cond=teacher_conditioning, cfg_scale=self.ode_warmup_cfg, dist_shift=self.teacher_model.dist_shift, batch_cfg=True)
            # if ode_step_divisor != self.ode_n_sample_steps, subsample x and t in diff_states to ode_step_divisor evenly spaced values
            if self.ode_step_divisor != self.ode_n_sampling_steps:
                self.diff_states['x'] = self.diff_states['x'][::self.ode_n_sampling_steps // self.ode_step_divisor]
                self.diff_states['t'] = self.diff_states['t'][::self.ode_n_sampling_steps // self.ode_step_divisor]
        conditioning = self.diffusion.conditioner(self.ode_metadata, self.device)

        if self.inpainting_config is not None:
            conditioning['inpaint_mask'] = [self.ode_inpaint_mask]
            conditioning['inpaint_masked_input'] = [self.ode_inpaint_masked_input]

        batch_size = diffusion_input.shape[0]
        channels = diffusion_input.shape[1]
        seq_len = diffusion_input.shape[2]

        if getattr(self, 'diffusion_forcing', False):
            if self.diffusion.model.model.causal and self.diffusion.model.model.causal > 1:
                block_size = self.diffusion.model.model.causal
            else:
                block_size = 1

            timestep = torch.randint(
                0,
                self.ode_step_divisor,
                [batch_size, seq_len],
                device=diffusion_input.device,
                dtype=torch.long
            )
            timestep = timestep.reshape(
                    timestep.shape[0], -1, block_size)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            ixs = timestep.reshape(timestep.shape[0], -1)
            t_all = self.diff_states['t'].clone().detach().squeeze(-1).to(self.device)  # (steps,)
            t = t_all[ixs]  # (batch, seq_len)
            t = t.to(self.device).detach().clone().requires_grad_(False)
            x_all_perm = self.diff_states['x'].clone().detach().permute(1, 0, 3, 2)  # (batch, steps, seq_len, channels
            x_t = torch.gather(
                x_all_perm, dim=1,
                index=ixs.reshape(batch_size, 1, seq_len, 1).expand(-1, -1, -1, channels).to(diffusion_input.device)
            ).squeeze(1)  
            x_t = x_t.permute(0, 2, 1)  # (batch, channels, seq_len)
            x_t = x_t.to(self.device).detach().clone().requires_grad_(False)


            # num_blocks = (seq_len + block_size - 1) // block_size  # Ceiling division

            

            # block_ixs = torch.randint(0, self.ode_n_sampling_steps, (batch_size, num_blocks), device=diffusion_input.device)

            # ixs = torch.repeat_interleave(block_ixs, block_size, dim=1)[:, :seq_len]

            # # self.diff_states['x']: (steps, batch, channels, seq_len)
            # # self.diff_states['t']: (steps, 1)
            # # We'll gather for each [b, p] the corresponding step
            # # For t: shape (steps, 1) -> (steps,)
            # t_all = self.diff_states['t'].clone().detach().squeeze(-1).to(self.device)  # (steps,)
            # # For x: (steps, batch, channels, seq_len)
            # x_all = self.diff_states['x'].clone().detach()  # (steps, batch, channels, seq_len)

            # # Prepare indices for advanced indexing
            # # For each [b, p], get step = ixs[b, p], batch = b, channel = : , position = p
            # b_idx = torch.arange(batch_size, device=diffusion_input.device).view(-1, 1).expand(-1, seq_len)  # (batch, seq_len)
            # p_idx = torch.arange(seq_len, device=diffusion_input.device).view(1, -1).expand(batch_size, -1)  # (batch, seq_len)

            # # Gather t: for each [b, p], t[ixs[b, p]]
            # t = t_all[ixs]  # (batch, seq_len)

            # # Gather x_t: for each [b, p], x_all[ixs[b, p], b, :, p]
            # # We'll use gather, but need to flatten indices
            # flat_ixs = ixs.flatten()  # (batch*seq_len,)
            # flat_b_idx = b_idx.flatten()
            # flat_p_idx = p_idx.flatten()
            # # x_all: (steps, batch, channels, seq_len)
            # x_t = x_all[flat_ixs, flat_b_idx, :, flat_p_idx]  # (batch*seq_len, channels)
            # # Reshape to (batch, seq_len, channels)
            # x_t = x_t.view(batch_size, seq_len, channels).permute(0, 2, 1)  # (batch, channels, seq_len)

            # t = t.to(self.device).detach().clone().requires_grad_(False)
            # x_t = x_t.to(self.device).detach().clone().requires_grad_(False)

            # The diffusion model may expect t as (batch, seq_len) or (batch, 1, seq_len)
            # We'll try (batch, seq_len)
            v_t_student = self.diffusion(x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob)
            denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))
            ode_mse_loss = F.mse_loss(denoised_student, self.diff_states['target'])
        else:
            # Old behavior: one index per batch
            ixs = torch.randint(0, self.ode_step_divisor, (diffusion_input.shape[0],))
            t = self.diff_states['t'].clone().detach()[ixs].squeeze(-1)
            x_t = self.diff_states['x'].clone().detach()[ixs.flatten(), torch.arange(diffusion_input.shape[0])].squeeze(0)

            t = t.to(self.device).detach().clone().requires_grad_(False)
            x_t = x_t.to(self.device).detach().clone().requires_grad_(False)

            v_t_student = self.diffusion(x_t, t, cond=conditioning, cfg_dropout_prob=self.cfg_dropout_prob)
            denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))
            ode_mse_loss = F.mse_loss(denoised_student, self.diff_states['target'])

        return ode_mse_loss

    def calculate_disc_loss(self, real_scores, fake_scores):
    
        # Relativistic discriminator loss
        diff = real_scores - fake_scores
        loss_dis = F.softplus(-diff).mean() * self.dis_gan_weight

        return loss_dis

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers
        else:
            sched_gen = sched_disc = None

        log_dict = {}

        if self.global_step < self.ode_warmup_steps:
            ode_mse_loss = self.ode_warmup_step(diffusion_input, metadata, padding_masks)

            opt_gen.zero_grad()
            self.manual_backward(ode_mse_loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict = {'train/ode_mse_loss': ode_mse_loss.detach()}
            self.log_dict(log_dict, prog_bar=True, on_step=True)

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            return ode_mse_loss

        # Start trimming after ODE warmup to avoid sequence length issues
        if self.trim_config is not None:
            if random.random() < self.trim_prob:
                
                data_lengths = (torch.sum(padding_masks, dim=1) - 1).tolist()
                
                if self.trim_type == "random_item":
                    trim_length = max(random.choice(data_lengths), 128)

                diffusion_input = diffusion_input[:, :, :trim_length]

        conditioning = self.diffusion.conditioner(metadata, self.device)

        if self.inpainting_config is not None:

            # Create a mask of random length for a random slice of the input
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=padding_masks, **self.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        gen_noise = torch.randn_like(diffusion_input)
        if self.diffusion_forcing:
            # Sample a different timestep for each batch and position

            if self.diffusion.model.model.causal and self.diffusion.model.model.causal > 1:
                block_size = self.diffusion.model.model.causal
            else:
                block_size = 1
            num_blocks = (diffusion_input.shape[2] + block_size - 1) // block_size
            block_t = self.gen_noise_dist((diffusion_input.shape[0], num_blocks)).to(self.device)
            t = torch.repeat_interleave(block_t, block_size, dim=1)[..., :diffusion_input.shape[2]]
            # t = self.gen_noise_dist((diffusion_input.shape[0], diffusion_input.shape[2])).to(self.device)
            x_t = diffusion_input * (1-t)[:, None] + gen_noise * t[:, None]
        else:
            t = self.gen_noise_dist(reals.shape[0])  
            x_t = diffusion_input * (1-t)[:, None, None] + gen_noise * t[:, None, None]       
      
        train_gen = (self.global_step % 2) == 0

        if train_gen or self.global_step < self.ode_warmup_steps:
            v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob)
        else:
            x_t = x_t.requires_grad_(False)
            with torch.no_grad():
                v_t_student = checkpoint(self.diffusion, x_t, t, cond=conditioning).detach()

        denoised_student = euler_step(x_t, v_t_student, t, torch.zeros_like(t))

        if type(self.discriminator_dit_layer) == list:
            # self.discriminator_dit_layer is a list of layers, randomly choose one
            discriminator_dit_layer = random.choice(self.discriminator_dit_layer)
        else:
            discriminator_dit_layer = self.discriminator_dit_layer

        if train_gen:
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

            if self.diffusion_ema is not None:
                self.diffusion_ema.update()

            # Get discriminator scores for adversarial loss
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            x_t_gan = denoised_student * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]

            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            v_t_gan_hidden_states = self.discriminator(x_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, use_checkpointing=True, exit_layer_ix=discriminator_dit_layer, cfg_dropout_prob = self.cfg_dropout_prob)

            disc_scores = self.discriminator_head(v_t_gan_hidden_states.transpose(1, 2))

            log_dict['gen_disc_scores_mean'] = disc_scores.mean().detach()
            log_dict['gen_disc_scores_std'] = disc_scores.std().detach()

            # Relativistic discriminator loss
            x_t_gan_real = diffusion_input * (1-t_gan)[:, None, None] + noise * t_gan[:, None, None]
            v_t_gan_real_hidden_states = self.discriminator(x_t_gan_real, t_gan, cond=fake_conditioning, cfg_scale=1.0, use_checkpointing=True, exit_layer_ix=discriminator_dit_layer, cfg_dropout_prob = self.cfg_dropout_prob)
            disc_scores_real = self.discriminator_head(v_t_gan_real_hidden_states.transpose(1, 2))

            diff = disc_scores_real - disc_scores

            loss_adv = F.softplus(diff).mean() * self.gen_gan_weight

            loss = loss_adv

            log_dict['train/adv_loss'] = loss_adv.detach()

            opt_gen.zero_grad()

            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                sched_gen.step()

            log_dict['train/gen_loss'] = loss.detach()
        else:
            denoised_student = denoised_student.detach().requires_grad_(True)
            t_gan = self.dis_noise_dist(reals.shape[0])
            noise = torch.randn_like(denoised_student)
            reals_t_gan = diffusion_input * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            denoised_t_gan = denoised_student * (1-t_gan)[..., None, None] + noise * t_gan[..., None, None]
            
            reals_t_gan = reals_t_gan.detach().requires_grad_(True)
            denoised_t_gan = denoised_t_gan.detach().requires_grad_(True)
            
            fake_conditioning = self.discriminator.conditioner(metadata, self.device)

            if self.inpainting_config is not None:
                fake_conditioning['inpaint_mask'] = [inpaint_mask]
                fake_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            reals_gan_hidden_states = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=discriminator_dit_layer, cfg_dropout_prob=self.cfg_dropout_prob).transpose(1,2)
            denoised_gan_hidden_states = checkpoint(self.discriminator, denoised_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=discriminator_dit_layer, cfg_dropout_prob=self.cfg_dropout_prob).transpose(1,2)

            disc_scores_reals = checkpoint(self.discriminator_head, reals_gan_hidden_states)
            disc_scores_denoised = checkpoint(self.discriminator_head, denoised_gan_hidden_states)

            if self.include_grad_penalties:
        
                r1_approx_variance = 0.05

                noised_reals_t_gan = reals_t_gan + r1_approx_variance * torch.randn_like(reals_t_gan)
                noised_denoised_t_gan = denoised_t_gan + r1_approx_variance * torch.randn_like(denoised_t_gan)
                noised_reals_gan_hidden_states = checkpoint(self.discriminator, noised_reals_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=discriminator_dit_layer).transpose(1,2)
                noised_denoised_gan_hidden_states = checkpoint(self.discriminator, noised_denoised_t_gan, t_gan, cond=fake_conditioning, cfg_scale=1.0, exit_layer_ix=discriminator_dit_layer).transpose(1,2)
                disc_scores_noised_reals = checkpoint(self.discriminator_head, noised_reals_gan_hidden_states)
                disc_scores_noised_denoised = checkpoint(self.discriminator_head, noised_denoised_gan_hidden_states)
                r1_diff = disc_scores_noised_reals - disc_scores_reals
                r2_diff = disc_scores_noised_denoised - disc_scores_denoised

                r1_penalty = torch.sum(r1_diff ** 2, dim=[1, 2])
                r2_penalty = torch.sum(r2_diff ** 2, dim=[1, 2])

                log_dict['r1_penalty'] = r1_penalty.mean().detach()
                log_dict['r2_penalty'] = r2_penalty.mean().detach()

                grad_penalty_loss = ((r1_penalty.mean() + r2_penalty.mean())/2)

                log_dict['train/grad_penalty_loss'] = grad_penalty_loss.detach()
            else:
                grad_penalty_loss = torch.tensor(0.0, device=self.device)

           

            log_dict['disc_real_scores_mean'] = disc_scores_reals.mean().detach()
            log_dict['disc_fake_scores_mean'] = disc_scores_denoised.mean().detach()
            log_dict['disc_real_scores_std'] = disc_scores_reals.std().detach()
            log_dict['disc_fake_scores_std'] = disc_scores_denoised.std().detach()

            loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_denoised)

            if self.do_contrastive_disc:

                rolled_metadata = []

                for i in range(reals.shape[0]):
                    rolled_keys = ["prompt"]
                    # rolled_keys = ["prompt", "pesto_pitch", "spectral_centroid", "rms_energy"]
                    rolled_metadata.append(metadata[i])
                    for rolled_key in rolled_keys:
                        rolled_metadata[i][rolled_key] = metadata[(i + 1) % reals.shape[0]][rolled_key]

                rolled_conditioning = self.discriminator.conditioner(rolled_metadata, self.device)

                if self.inpainting_config is not None:
                    # Hold inpainting conditioning constant during contrastive conditioning
                    rolled_conditioning['inpaint_mask'] = [inpaint_mask]
                    rolled_conditioning['inpaint_masked_input'] = [inpaint_masked_input]

                rolled_reals_gan_hidden_states = checkpoint(self.discriminator, reals_t_gan, t_gan, cond=rolled_conditioning, cfg_scale=1.0, exit_layer_ix=discriminator_dit_layer, cfg_dropout_prob = self.cfg_dropout_prob).transpose(1,2)

                disc_scores_rolled_reals = checkpoint(self.discriminator_head, rolled_reals_gan_hidden_states)

                contrastive_loss_dis = self.calculate_disc_loss(disc_scores_reals, disc_scores_rolled_reals) * self.contrastive_loss_weight

                log_dict['train/contrastive_loss_dis'] = contrastive_loss_dis.detach()
            else:
                contrastive_loss_dis = 0

            loss = loss_dis + contrastive_loss_dis + grad_penalty_loss

            log_dict['train/dis_loss'] = loss_dis.detach()

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']
            log_dict['train/discriminator_loss'] = loss.detach()
            opt_disc.zero_grad()

            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(list(self.discriminator_head.parameters()) + list(self.discriminator.parameters()), self.clip_grad_norm)
            opt_disc.step()

            if sched_disc is not None:
                sched_disc.step()

        log_dict['train/std_data'] = diffusion_input.std()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        
        return loss

    def build_noise_dist(self, key):
        dist = self.noise_dist_config.get(key, 'uniform')
        match dist:
            case 'uniform':
                return lambda b: self.rng.draw(b)[:, 0].to(self.device)
            case 'logit_normal':
                return lambda b: torch.sigmoid(torch.randn(b, device=self.device))
            case 'trunc_logit_normal':
                return lambda b: 1 - truncated_logistic_normal_rescaled(b).to(self.device)
            case 'one_shot':
                return lambda b: torch.ones(b, device=self.device)
            case 'denoised':
                return lambda b: torch.zeros(b, device=self.device)
            case 'logsnr_uniform':
                
                min_logsnr = -6
                max_logsnr = 2

                return lambda b: torch.sigmoid(-(torch.rand(b, device=self.device) * (max_logsnr - min_logsnr) + min_logsnr))
            case _:
                raise ValueError(f"Invalid noise distribution: {dist}")