import logging
from typing import Any, Mapping
import sys

import einops
import torch
from torch.nn import functional as F

sys.path.insert(0, '.')  # nopep8
from utils.utils import instantiate_from_config
from model.modules.transformer import Block, Config

def init_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Synchformer(torch.nn.Module):
    ''' The module has similar structure to SparseSync (SparseSync) but has a diffrerent
    forward pass. It expects the output of the feature extractors to have global and
    segment-level representations.'''

    def __init__(self, afeat_extractor, vfeat_extractor, aproj, vproj, transformer):
        super().__init__()
        self.vfeat_extractor = instantiate_from_config(vfeat_extractor)
        self.afeat_extractor = instantiate_from_config(afeat_extractor)
        # bridging the s3d latent dim (1024) into what is specified in the config
        # to match e.g. the transformer dim
        self.vproj = instantiate_from_config(vproj)
        self.aproj = instantiate_from_config(aproj)
        self.transformer = instantiate_from_config(transformer)

    def forward(self, vis: torch.Tensor, aud: torch.Tensor, targets: torch.Tensor = None, for_loop=False,
                vis_mask: torch.Tensor = None, aud_mask: torch.Tensor = None, loss_fn=None):
        '''
        Args:
            vis (torch.Tensor): RGB frames (B, S, Tv, C, H, W)
            aud (torch.Tensor): audio spectrograms (B, S, 1, F, Ta)
            for_loop (bool): if True, will use a for loop inside of feat_extractors to iterate over the
                             segments or process them in parallel (False), treating segment dim as batch dim
                             (speed-memory tradeoff).
            vis_mask (torch.Tensor): mask for the visual tokens (as input)
            aud_mask (torch.Tensor): mask for the audio tokens (as input)
        Returns:
            tuple(Tensor, Tensor), Tensor: loss values, logits
        '''
        vis = self.extract_vfeats(vis, for_loop, vis_mask=vis_mask)
        aud = self.extract_afeats(aud, for_loop, aud_mask=aud_mask)

        vis = self.vproj(vis)
        aud = self.aproj(aud)

        # flatten the segment dim (treating the sequence of segments as a single sequence)
        B, S, tv, D = vis.shape
        B, S, ta, D = aud.shape
        vis = vis.view(B, S*tv, D)  # (B, S*tv, D)
        aud = aud.view(B, S*ta, D)  # (B, S*ta, D)

        # self.transformer will concatenate the vis and aud in one sequence with aux tokens,
        # ie `CvvvvMaaaaaa`, and will return the logits for the CLS tokens
        logits = self.transformer(vis, aud)  # (B, cls); or (B, cls) and (B, 2) if DoubtingTransformer

        loss = self.compute_loss(logits, targets, loss_fn)  # (B,); or a tuple of (B,) and (B,)

        return loss, logits

    def extract_vfeats(self, vis, for_loop, vis_mask=None):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        if vis_mask is not None:
            vis_mask = vis_mask.permute(0, 1, 3, 2, 4, 5)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis, _ = self.vfeat_extractor(vis, for_loop=for_loop, cont_mask=vis_mask)
        return vis

    def extract_afeats(self, aud, for_loop, aud_mask=None):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        if aud_mask is not None:
            aud_mask = aud_mask.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # (B, S, ta, D), e.g. (B, 7, 6, 768)
        aud, _ = self.afeat_extractor(aud, for_loop=for_loop, cont_mask=aud_mask)
        return aud

    '''def compute_loss(self, logits, targets, loss_fn: str = None):
        loss = None
        if targets is not None:
            if loss_fn is None or loss_fn == 'cross_entropy':
                # logits: (B, cls) and targets: (B,)ug
                loss = F.cross_entropy(logits, targets)
            else:
                raise NotImplementedError(f'Loss {loss_fn} not implemented')
        return loss'''
    
    def compute_loss(self, logits, targets, loss_fn: str = None):
        loss = None
        if targets is not None:
            targets = targets.to(dtype=torch.float32)
            targets = targets.to(logits.device)
            if loss_fn is None or loss_fn == 'mse':
                # logits: (B, cls) and targets: (B,)
                print("here compute_loss")
                print(logits.view(-1), targets)
                loss = F.mse_loss(logits.view(-1), targets)
            else:
                raise NotImplementedError(f'Loss {loss_fn} not implemented')
        return loss

    # def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
    #     ''' Overriding the default load_state_dict to allow loading a state dict with longer sequence.'''
    #     model_sd = self.state_dict()
    #     initialized_keys = []
    #     skipped_keys = []

    #     for key, param in model_sd.items():
    #         if key in sd:
    #             if param.shape == sd[key].shape:
    #                 # Load the parameter if the shape matches
    #                 model_sd[key] = sd[key]
    #                 initialized_keys.append(key)
    #             else:
    #                 # Randomly initialize if the shape does not match
    #                 skipped_keys.append(key)
    #                 logging.warning(f"Shape mismatch for key '{key}': "
    #                                 f"model expects {param.shape}, but got {sd[key].shape}. Randomly reinitializing.")
    #                 if param.requires_grad:
    #                     model_sd[key] = torch.nn.init.normal_(param)
    #         else:
    #             # Randomly initialize if the key is not in the state dict
    #             skipped_keys.append(key)
    #             logging.warning(f"Key '{key}' not found in the state dict. Randomly initializing.")
    #             if param.requires_grad:
    #                 model_sd[key] = torch.nn.init.normal_(param)

    #     # Log results
    #     logging.info(f"Initialized {len(initialized_keys)} parameters from the state dict.")
    #     logging.info(f"Randomly initialized {len(skipped_keys)} parameters.")
    #     # Handle size mismatches for 'off_head.weight' and 'off_head.bias'
    #     keys_to_resize = ['transformer.off_head.weight', 'transformer.off_head.bias']
    #     for key in keys_to_resize:
    #         if key in sd:
    #             expected_shape = getattr(self.transformer.off_head, key.split('.')[-1]).shape  # Get the expected shape
    #         current_shape = sd[key].shape
    #         if current_shape != expected_shape:  # Only handle mismatches
    #             logging.warning(f"Resizing {key} from {current_shape} to {expected_shape}")
                
    #             if 'weight' in key:
    #                 # Initialize a new weight tensor with random values
    #                 new_weights = torch.randn(expected_shape, device=sd[key].device, dtype=sd[key].dtype)
    #                 # Copy the existing weights into the new tensor
    #                 #new_weights[:current_shape[0], :current_shape[1]] = sd[key]
    #                 sd[key] = new_weights

    #             elif 'bias' in key:
    #                 # Initialize a new bias tensor with random values
    #                 new_bias = torch.randn(expected_shape, device=sd[key].device, dtype=sd[key].dtype)
    #                 # Copy the existing bias into the new tensor
    #                 #new_bias[:current_shape[0]] = sd[key]
    #                 sd[key] = new_bias
    #     return super().load_state_dict(sd, strict=False)
    
    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        ''' Overriding the default load_state_dict to allow loading a state dict with longer sequence.'''
        if 'transformer.pos_emb_cfg.pos_emb' in sd:
            # get the weight length from the state dict
            weight_len = sd['transformer.pos_emb_cfg.pos_emb'].shape[1]
            # get the weight length from the current model
            self_len = self.transformer.pos_emb_cfg.pos_emb.shape[1]
            # trim the weights if the state dict is longer than the current model
            if weight_len > self_len:
                sd['transformer.pos_emb_cfg.pos_emb'] = sd['transformer.pos_emb_cfg.pos_emb'][:, :self_len, :]
                logging.warning(f'Trimming the state dict for pos emb from {weight_len} to {self_len}')
            elif weight_len < self_len:
                raise ValueError(f'Cant load state dict with shorter seq len ({weight_len} vs {self_len})')
        print("yes")
        # Handle size mismatches for 'off_head.weight' and 'off_head.bias'
        keys_to_resize = ['transformer.off_head.weight', 'transformer.off_head.bias']
        for key in keys_to_resize:
            if key in sd:
                expected_shape = getattr(self.transformer.off_head, key.split('.')[-1]).shape  # Get the expected shape
            current_shape = sd[key].shape
            if current_shape != expected_shape:  # Only handle mismatches
                logging.warning(f"Resizing {key} from {current_shape} to {expected_shape}")
                
                if 'weight' in key:
                    # Initialize a new weight tensor with random values
                    new_weights = torch.randn(expected_shape, device=sd[key].device, dtype=sd[key].dtype)
                    # Copy the existing weights into the new tensor
                    #new_weights[:current_shape[0], :current_shape[1]] = sd[key]
                    sd[key] = new_weights

                elif 'bias' in key:
                    # Initialize a new bias tensor with random values
                    new_bias = torch.randn(expected_shape, device=sd[key].device, dtype=sd[key].dtype)
                    # Copy the existing bias into the new tensor
                    #new_bias[:current_shape[0]] = sd[key]
                    sd[key] = new_bias
        return super().load_state_dict(sd, strict=False)

'''class GlobalTransformer(torch.nn.Module):
    #Same as in SparseSync but without the selector transformers and the head

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd, pos_emb_cfg=None,
                 pos_emb_cfg_v=None, pos_emb_cfg_a=None, off_head_cfg=None) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        self.vis_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)

        # aux tokens
        self.OFF_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))

        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = torch.nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout1d(tok_pdrop)
        
        if pos_emb_cfg is not None:
            self.pos_emb_cfg = instantiate_from_config(pos_emb_cfg)
        if pos_emb_cfg_v is not None:
            self.pos_emb_cfg_v = instantiate_from_config(pos_emb_cfg_v)
        if pos_emb_cfg_a is not None:
            self.pos_emb_cfg_a = instantiate_from_config(pos_emb_cfg_a)

        # Cross-attention layers
        self.audio_query_video = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True) for _ in range(self.config.n_layer)])
        self.video_query_audio = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True) for _ in range(self.config.n_layer)])

        # Dropout
        self.cross_attn_dropout = torch.nn.Dropout(embd_pdrop)
        
        # LayerNorm
        self.audio_ln = torch.nn.LayerNorm(n_embd)
        self.video_ln = torch.nn.LayerNorm(n_embd)

        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks_a = torch.nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # pre-output norm
        self.ln_f_a = torch.nn.LayerNorm(self.config.n_embd)
        
        self.blocks_v = torch.nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # pre-output norm
        self.ln_f_v = torch.nn.LayerNorm(self.config.n_embd)
        
        # maybe add a head
        if off_head_cfg is not None:
            self.off_head = instantiate_from_config(off_head_cfg)
            # Define trainable alpha parameter
            self.layer1 = torch.nn.Linear(256, 64)
            self.act1 = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(64, 1)
            self.alpha = torch.nn.Parameter(torch.tensor(1.0))  # Initialize with 1.0

        self.apply(init_weights)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        B, Sv, D = v.shape
        B, Sa, D = a.shape

        # Broadcasting special tokens to batch size
        off_tok = einops.repeat(self.OFF_tok, '1 1 d -> b 1 d', b=B)
        mod_tok = einops.repeat(self.MOD_tok, '1 1 d -> b 1 d', b=B)

        v = self.vis_in_lnorm(v)
        a = self.aud_in_lnorm(a)
        if self.tok_pdrop > 0:
            v = self.tok_drop_vis(v)
            a = self.tok_drop_aud(a)
        if hasattr(self, 'pos_emb_cfg_v'):
            v = self.pos_emb_cfg_v(v)
        if hasattr(self, 'pos_emb_cfg_a'):
            a = self.pos_emb_cfg_a(a)

        # Cross-attention
        a_residual = a
        a = self.cross_attn_dropout(a)  # Apply Dropout
        for i in range(self.config.n_layer):
            audio_out, _ = self.audio_query_video[i](a, v, v)  # Apply current MultiheadAttention layer
            a = audio_out  # Update the input for the next layer (if needed)
        #print(f"nivi2 audio_out.shape: {audio_out.shape}")
        audio_out = self.audio_ln(audio_out)  # Apply LayerNorm
        #audio_out = audio_out + a_residual

        # Cross-attention: Video queries Audio
        v_residual = v
        v = self.cross_attn_dropout(v)  # Apply Dropout
        # Apply each layer in the ModuleList for video -> audio attention
        for i in range(self.config.n_layer):
            video_out, _ = self.video_query_audio[i](v, a, a)  # Apply current MultiheadAttention layer
            v = video_out  # Update the input for the next layer (if needed)
        #print(f"nivi2 video_out.shape: {video_out.shape}")
        video_out = self.video_ln(video_out)  # Apply LayerNorm
        #video_out = video_out + v_residual


        # Self-attention for audio
        audio_out = self.drop(audio_out)
        audio_out = self.blocks_a(audio_out)
        audio_out = self.ln_f_a(audio_out)

        # Self-attention for video
        video_out = self.drop(video_out)
        video_out = self.blocks_v(video_out)
        video_out = self.ln_f_v(video_out)

        # Self-attention
        # Concatenate special tokens, video, and audio
        x = torch.cat((video_out, audio_out), dim=1)
        x = torch.mean(x, dim=1)
        #print(f"nivi: {x.shape}")
        #print(f"nivi3 x concat shape: {x.shape}")
        # Apply positional embeddings for concatenated input if they exist
        # if hasattr(self, 'pos_emb_cfg') and self.pos_emb_cfg:
        #     x = self.pos_emb_cfg(x)

        # # Self-attention: Dropout -> Stem -> Norm
        # x = self.drop(x)  # Apply Dropout
        # x = self.blocks(x)  # Apply Transformer blocks (Self-attention)
        # x = self.ln_f(x)  # Apply final LayerNorm
        #print(f"nivi4 x Tx output shape: {x.shape}")
        # maybe add heads
        if attempt_to_apply_heads and hasattr(self, 'off_head'):
            #x = self.off_head(x[:,0,:])
            x = self.off_head(x)
            #Apply the tanh activation
            x = self.layer1(x)
            x = self.act1(x)
            x = self.layer2(x)
            x = torch.tanh(self.alpha * x)
            #print(f"nivi5 x mlp output shape: {x.shape}")
        return x*0.2
        #return x'''

class GlobalTransformer(torch.nn.Module):
    '''Same as in SparseSync but without the selector transformers and the head'''

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd, pos_emb_cfg=None,
                 pos_emb_cfg_v=None, pos_emb_cfg_a=None, off_head_cfg=None) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        self.vis_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(self.config.n_embd)

        # aux tokens
        self.OFF_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))

        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = torch.nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout1d(tok_pdrop)
        
        if pos_emb_cfg is not None:
            self.pos_emb_cfg = instantiate_from_config(pos_emb_cfg)
        if pos_emb_cfg_v is not None:
            self.pos_emb_cfg_v = instantiate_from_config(pos_emb_cfg_v)
        if pos_emb_cfg_a is not None:
            self.pos_emb_cfg_a = instantiate_from_config(pos_emb_cfg_a)

        # Cross-attention layers
        #self.audio_query_video = torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True)
        #self.video_query_audio = torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True)
        self.audio_query_video = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True) for _ in range(self.config.n_layer)])
        self.video_query_audio = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True) for _ in range(self.config.n_layer)])

        # Dropout
        self.cross_attn_dropout = torch.nn.Dropout(embd_pdrop)
        
        # LayerNorm
        self.audio_ln = torch.nn.LayerNorm(n_embd)
        self.video_ln = torch.nn.LayerNorm(n_embd)

        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks = torch.nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # pre-output norm
        self.ln_f = torch.nn.LayerNorm(self.config.n_embd)
        
        # maybe add a head
        if off_head_cfg is not None:
            self.off_head = instantiate_from_config(off_head_cfg)
            # Define trainable alpha parameter
            self.layer1 = torch.nn.Linear(256, 64)
            self.act1 = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(64, 1)
            self.alpha = torch.nn.Parameter(torch.tensor(1.0))  # Initialize with 1.0

        self.apply(init_weights)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        B, Sv, D = v.shape
        B, Sa, D = a.shape

        # Broadcasting special tokens to batch size
        off_tok = einops.repeat(self.OFF_tok, '1 1 d -> b 1 d', b=B)
        mod_tok = einops.repeat(self.MOD_tok, '1 1 d -> b 1 d', b=B)

        v = self.vis_in_lnorm(v)
        a = self.aud_in_lnorm(a)
        if self.tok_pdrop > 0:
            v = self.tok_drop_vis(v)
            a = self.tok_drop_aud(a)
        if hasattr(self, 'pos_emb_cfg_v'):
            v = self.pos_emb_cfg_v(v)
        if hasattr(self, 'pos_emb_cfg_a'):
            a = self.pos_emb_cfg_a(a)

        # Cross-attention
        a_residual = a
        a = self.cross_attn_dropout(a)  # Apply Dropout
        for i in range(self.config.n_layer):
            audio_out, _ = self.audio_query_video[i](a, v, v)  # Apply current MultiheadAttention layer
            a = audio_out  # Update the input for the next layer (if needed)
        #print(f"nivi2 audio_out.shape: {audio_out.shape}")
        audio_out = self.audio_ln(audio_out)  # Apply LayerNorm
        audio_out = audio_out + a_residual

        # Cross-attention: Video queries Audio
        v_residual = v
        v = self.cross_attn_dropout(v)  # Apply Dropout
        # Apply each layer in the ModuleList for video -> audio attention
        for i in range(self.config.n_layer):
            video_out, _ = self.video_query_audio[i](v, a, a)  # Apply current MultiheadAttention layer
            v = video_out  # Update the input for the next layer (if needed)
        #print(f"nivi2 video_out.shape: {video_out.shape}")
        video_out = self.video_ln(video_out)  # Apply LayerNorm
        video_out = video_out + v_residual

        # Self-attention
        # Concatenate special tokens, video, and audio

        #video_out, audio_out = v, a
        x = torch.cat((off_tok, video_out, mod_tok, audio_out), dim=1)
        #print(f"nivi3 x concat shape: {x.shape}")
        # Apply positional embeddings for concatenated input if they exist
        if hasattr(self, 'pos_emb_cfg') and self.pos_emb_cfg:
            x = self.pos_emb_cfg(x)

        # Self-attention: Dropout -> Stem -> Norm
        x = self.drop(x)  # Apply Dropout
        x = self.blocks(x)  # Apply Transformer blocks (Self-attention)
        x = self.ln_f(x)  # Apply final LayerNorm
        #print(f"nivi4 x Tx output shape: {x.shape}")
        # maybe add heads
        if attempt_to_apply_heads and hasattr(self, 'off_head'):
            x = self.off_head(x[:,0,:])
            #Apply the tanh activation
            x = self.layer1(x)
            x = self.act1(x)
            x = self.layer2(x)
            x = torch.tanh(self.alpha * x)
            #print(f"nivi5 x mlp output shape: {x.shape}")
        return x*0.2
        #return x


class GlobalTransformerWithSyncabilityHead(GlobalTransformer):

    def __init__(self, tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd,
                 pos_emb_cfg=None, off_head_cfg=None) -> None:
        super().__init__(tok_pdrop, embd_pdrop, resid_pdrop, attn_pdrop, n_layer, n_head, n_embd, pos_emb_cfg,
                         off_head_cfg)
        # remove the off_head from the parent class
        self.off_head = torch.nn.Identity()  # this class is used only during ftuning so this is not needed
        self.sync_head = torch.nn.Linear(self.config.n_embd, 2)
        self.apply(init_weights)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        x = super().forward(v, a, targets, attempt_to_apply_heads=False)
        logits_sync = self.sync_head(x[:, 0, :])
        return logits_sync


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from omegaconf import OmegaConf
    from time import time

    cfg = OmegaConf.load('./configs/sync.yaml')
    cfg.training.use_half_precision = use_half_precision = False

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = instantiate_from_config(cfg.model)
    model = model.to(device)

    start_time = time()
    for i in range(3):
        vis = torch.rand(1, 125, 3, 224, 224, device=device)
        aud = torch.rand(1, 1, 257, 626, device=device)
        # cls_logits, off_logits, sync_logits = model(vis, aud)
        # inference in half precision
        with torch.cuda.amp.autocast(cfg.training.use_half_precision):
            out = model(vis, aud)
    print('Time:', round(time() - start_time, 3))
