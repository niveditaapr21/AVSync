import argparse
import subprocess
from pathlib import Path

import torch
import torchaudio
import torchvision
from omegaconf import OmegaConf

from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid, quantize_offset
from utils.utils import check_if_file_exists_else_download, which_ffmpeg
from scripts.train_utils import get_model, get_transforms, prepare_inputs


def reencode_video(path, vfps=25, afps=16000, in_size=256):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4'
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = f'{which_ffmpeg()}'
    # no info/error printing
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {path}'
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f' {new_path}'
    subprocess.call(cmd.split())
    cmd = f'{which_ffmpeg()}'
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {new_path}'
    cmd += f' -acodec pcm_s16le -ac 1'
    cmd += f' {new_path.replace(".mp4", ".wav")}'
    subprocess.call(cmd.split())
    return new_path

def decode_single_video_prediction(off_logits, grid, item, output_filename):
    print('Prediction Result:')
    # Redirect print output to a file
                    
    off_probs = {'off_logits':off_logits.item()}
    print(f"prediction: {off_probs['off_logits']}")
    with open(output_filename, 'a') as f:
        f.write(f"{off_probs['off_logits']}\n")
    return off_probs

def reconstruct_video_from_input(aud, vid, meta, orig_vid_path, v_start_i_sec, offset_sec, vfps, afps):
    raise NotImplementedError
    # assumptions
    n_fft = 512
    hop_length = 128
    torchvision_means = [0.485, 0.456, 0.406]
    torchvision_stds = [0.229, 0.224, 0.225]

    # inverse audio transforms
    assert aud.shape[0] == 1, f'batchsize > 1: imgs.shape {aud.shape}'
    means = meta['spec_means'].view(1, 1, -1, 1)
    stds = meta['spec_stds'].view(1, 1, -1, 1)
    spec = aud.cpu() * stds + means
    spec = spec.squeeze(0).squeeze(0)  # was: (B=1, C=1, F, Ta)
    # spec = torch.exp(spec)
    # AudioSpectrogram
    aud_rec = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)(torch.exp(spec))
    aud_rec = aud_rec[None]

    # inverse visual transforms
    means = torch.tensor(torchvision_means).view(1, 1, 3, 1, 1)
    stds = torch.tensor(torchvision_stds).view(1, 1, 3, 1, 1)
    vid_rec = ((vid.cpu() * stds + means) * 255).short()
    vid_rec = vid_rec[0].permute(0, 2, 3, 1)

    # make a path to the reconstructed video:
    vis_folder = Path.cwd() / 'vis'
    vis_folder.mkdir(exist_ok=True)
    save_vid_path = vis_folder / f'rec_{Path(orig_vid_path).stem}_off{offset_sec}_t{v_start_i_sec}.mp4'
    save_vid_path = str(save_vid_path)
    print(f'Reconstructed video: {save_vid_path} (vid_crop starts at {v_start_i_sec}, offset {offset_sec})')

    # save the reconstructed input
    torchvision.io.write_video(save_vid_path, vid_rec, vfps, audio_array=aud_rec, audio_fps=afps, audio_codec='aac')

def patch_config(cfg):
    # the FE ckpts are already in the model ckpt
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    # old checkpoints have different names
    cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
    return cfg

import os
import argparse
import torch
import torchvision
from omegaconf import OmegaConf

def main(args):
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Constants
    vfps = 25
    afps = 16000
    in_size = 256

    # Decode and save results
    output_filename = 'results.txt'

    with open(output_filename, 'w') as f:
        f.write(f"video_name,offset_set,prediction\n")

    # Process each video in the input directory
    for filename in os.listdir(args.input_dir):
        # Check if the file is a video
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        
        vid_path = os.path.join(args.input_dir, filename)
        
        # Construct paths for config and checkpoint
        cfg_path = f'./logs/sync_models/{args.exp_name}/cfg-{args.exp_name}.yaml'
        ckpt_path = f'./logs/sync_models/{args.exp_name}/{args.exp_name}.pt'

        # Check and download config and checkpoint if needed
        check_if_file_exists_else_download(cfg_path)
        check_if_file_exists_else_download(ckpt_path)

        # Load config
        cfg = OmegaConf.load(cfg_path)
        cfg = patch_config(cfg)

        # Check and reencode video if necessary
        print(f'Processing video: {filename}')
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        
        if info['video_fps'] != vfps or info['audio_fps'] != afps or min(H, W) != in_size:
            print(f'Reencoding {filename}. vfps: {info["video_fps"]} -> {vfps};', end=' ')
            print(f'afps: {info["audio_fps"]} -> {afps};', end=' ')
            print(f'{(H, W)} -> min(H, W)={in_size}')
            vid_path = reencode_video(vid_path, vfps, afps, in_size)
        else:
            print(f'Skipping reencoding for {filename}')

        device = torch.device(args.device)

        # Load the model
        _, model = get_model(cfg, device)
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        model.eval()

        # Process video
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)

        # Prepare video processing parameters
        max_off_sec = cfg.data.max_off_sec
        num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
        grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

        # Process with different offsets
        for v_start_i_sec in [0.0]:  # You can expand this to try multiple start times
            for offset_sec in [0.0]:  # You can expand this to try multiple offsets
                item = dict(
                    video=rgb, audio=audio, meta=meta, path=vid_path, split='test',
                    targets={'v_start_i_sec': v_start_i_sec, 'offset_sec': offset_sec},
                )

                # Check offset is within grid
                if not (min(grid) <= item['targets']['offset_sec'] <= max(grid)):
                    print(f'WARNING: offset_sec={item["targets"]["offset_sec"]} is outside the trained grid: {grid}')
                    continue

                # Apply transforms
                item = get_transforms(cfg, ['test'])['test'](item)

                # Prepare inputs for inference
                batch = torch.utils.data.default_collate([item])
                aud, vid, targets = prepare_inputs(batch, device)

                # Forward pass
                with torch.set_grad_enabled(False):
                    with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                        _, logits = model(vid, aud)

                
                
                # Redirect print output to a file
                with open(output_filename, 'a') as f:
                    f.write(f"{filename},{offset_sec},")
                    
                    # Use the custom print function to redirect output
                decode_single_video_prediction(logits, grid, item, output_filename)

        print(f'Finished processing {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch process videos in a directory')
    parser.add_argument('--exp_name', required=True, help='Experiment name in format: xx-xx-xxTxx-xx-xx')
    parser.add_argument('--input_dir', required=True, help='Directory containing input videos')
    parser.add_argument('--output_dir', required=True, help='Directory to save output results')
    parser.add_argument('--offset_sec', type=float, default=0.0, help='Default offset in seconds')
    parser.add_argument('--v_start_i_sec', type=float, default=0.0, help='Default video start time in seconds')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference on')
    args = parser.parse_args()
    
    main(args)