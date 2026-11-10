import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time  # Added for timing

# Import necessary tools from your project structure
from feeders import tools

# ==========================================
# 1. NTU RGB+D Bone Definitions
# ==========================================
ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
)


# ==========================================
# 2. Skeleton File Parser
# ==========================================
def read_skeleton_file(file_path):
    """
    Parses a standard NTU RGB+D .skeleton file.
    Returns:
        numpy array: (C, T, V, M) -> (3, Frames, 25, 2)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Skeleton file not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    line_idx = 0
    frame_count = int(lines[line_idx]);
    line_idx += 1

    # Initialize container: 3 channels (x,y,z), T frames, 25 joints, max 2 persons
    skeleton_data = np.zeros((3, frame_count, 25, 2))

    for t in range(frame_count):
        body_count = int(lines[line_idx]);
        line_idx += 1

        for m in range(body_count):
            line_idx += 1
            joint_count = int(lines[line_idx]);
            line_idx += 1

            if m < 2:
                for v in range(joint_count):
                    joint_line = lines[line_idx].strip().split()
                    skeleton_data[0, t, v, m] = float(joint_line[0])
                    skeleton_data[1, t, v, m] = float(joint_line[1])
                    skeleton_data[2, t, v, m] = float(joint_line[2])
                    line_idx += 1
            else:
                line_idx += joint_count

    return skeleton_data


# ==========================================
# 3. Preprocessing Pipeline
# ==========================================
def preprocess_chunk(data_chunk, model_window_size=64, use_bone=False):
    """
    Args:
        data_chunk: (C, T_raw, V, M)
        model_window_size: 64
        use_bone: bool
    Returns:
        processed_tensor: (C, 64, V, M)
    """
    # 1. Center this specific window
    origin = data_chunk[:, 0:1, 0:1, 0:1]
    data = data_chunk - origin

    # 2. Resize
    valid_frame_num = data.shape[1]
    data = tools.valid_crop_resize(data, valid_frame_num, [0.95], model_window_size)

    # 3. Bone Conversion
    if use_bone:
        bone_data = np.zeros_like(data)
        for v1, v2 in ntu_pairs:
            v1_idx, v2_idx = v1 - 1, v2 - 1
            bone_data[:, :, v1_idx, :] = data[:, :, v1_idx, :] - data[:, :, v2_idx, :]

        bone_data[:, :, 20, :] = data[:, :, 20, :]
        data = bone_data

    return data


# ==========================================
# 4. Utilities
# ==========================================
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load_model(config_path, checkpoint_path, device_id):
    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    Model = import_class(args['model'])
    model = Model(**args['model_args'])

    weights = torch.load(checkpoint_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device, args


# ==========================================
# 5. Sliding Window Inference with Timing
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--skeleton_path', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--window_width', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1)

    args = parser.parse_args()

    # Load Model
    model, device, config_args = load_model(args.config, args.checkpoint, args.device)
    use_bone = config_args['train_feeder_args'].get('bone', False)
    model_window_size = config_args['train_feeder_args'].get('window_size', 64)
    print(f"Modality: {'Bone' if use_bone else 'Joint'} | Model Input Size: {model_window_size}")

    # Load Data
    full_data = read_skeleton_file(args.skeleton_path)
    total_frames = full_data.shape[1]

    window_width = args.window_width if args.window_width > 0 else total_frames
    stride = args.stride if args.window_width > 0 else total_frames

    print(f"\nRunning Sliding Window: Total Frames={total_frames}, Width={window_width}, Stride={stride}")
    print(
        "==================================================================================================================")
    print(f"{'Window (Frames)':<20} | {'Top Prediction':<25} | {'Score':<8} | {'Preproc(ms)':<12} | {'Infer(ms)':<12}")
    print(
        "------------------------------------------------------------------------------------------------------------------")

    # Warmup GPU (Optional but recommended for accurate timing)
    if torch.cuda.is_available():
        dummy_input = torch.randn(1, 3, model_window_size, 25, 2).to(device)
        dummy_label = torch.zeros(1).long().to(device)
        model(dummy_input, dummy_label, dummy_input)
        torch.cuda.synchronize()

    for start_frame in range(0, total_frames, stride):
        end_frame = start_frame + window_width

        if end_frame > total_frames:
            if total_frames - start_frame < 20:
                break
            end_frame = total_frames

        raw_chunk = full_data[:, start_frame:end_frame, :, :]

        # ---------------- TIMING PREPROCESSING ----------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_start_prep = time.time()

        # 1. Preprocess Main Data
        input_data = preprocess_chunk(raw_chunk, model_window_size, use_bone)
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(device)

        # 2. Preprocess Joint Data (Required for BlockGCN/HierarchicalGCN signature)
        processed_joint = preprocess_chunk(raw_chunk, model_window_size, use_bone=False)
        processed_joint = torch.from_numpy(processed_joint).float().unsqueeze(0).to(device)
        dummy_label = torch.zeros(1).long().to(device)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_end_prep = time.time()
        prep_ms = (t_end_prep - t_start_prep) * 1000
        # ------------------------------------------------------

        # ---------------- TIMING INFERENCE --------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_start_inf = time.time()

        with torch.no_grad():
            output, _ = model(input_tensor, dummy_label, processed_joint)
            probs = F.softmax(output, dim=1)
            top_val, top_idx = torch.topk(probs, 3)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_end_inf = time.time()
        inf_ms = (t_end_inf - t_start_inf) * 1000
        # ------------------------------------------------------

        # Result
        top_cls = top_idx[0, 0].item()
        top_score = top_val[0, 0].item()
        win_str = f"[{start_frame}:{end_frame}]"

        print(
            f"{win_str:<20} | Class {top_cls:<3} ({top_score:.2f})       | {top_score:.4f}   | {prep_ms:.2f}         | {inf_ms:.2f}")

        if end_frame == total_frames:
            break

    print(
        "==================================================================================================================")