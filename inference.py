import argparse
import yaml
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Add current directory to path
sys.path.append(os.getcwd())


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def read_skeleton_file(file_path):
    """
    Parses a standard NTU RGB+D .skeleton file.
    Returns numpy array of shape (C, T, V, M) -> (3, T, 25, 2)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Skeleton file not found: {file_path}")

    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()

    if len(datas) < 1:
        return np.zeros((3, 1, 25, 2))  # Empty return

    frame_count = int(datas[0])
    # NTU constants
    max_body = 2
    num_joint = 25

    # Storage: (M, T, V, C) -> converted to (C, T, V, M) later
    # Initialize with zeros
    data = np.zeros((max_body, frame_count, num_joint, 3))

    current_line = 1
    for t in range(frame_count):
        if current_line >= len(datas):
            break
        body_count = int(datas[current_line])
        current_line += 1

        for m in range(body_count):
            if current_line >= len(datas):
                break
            # Skip body info line (tracking ID, etc.)
            current_line += 1

            # Read 25 joints
            num_joint_lines = int(datas[current_line])  # usually 25
            current_line += 1

            # Only process up to max_body (2)
            if m < max_body:
                for j in range(num_joint):
                    if current_line >= len(datas):
                        break
                    line_content = datas[current_line].split()
                    # x, y, z are the first 3 elements
                    data[m, t, j, 0] = float(line_content[0])
                    data[m, t, j, 1] = float(line_content[1])
                    data[m, t, j, 2] = float(line_content[2])
                    current_line += 1
            else:
                # Skip remaining joints for extra bodies
                current_line += num_joint

    # Transpose to (C, T, V, M)
    # Current: (M, T, V, C) -> (C, T, V, M)
    # Permute: (3, 1, 2, 0)
    data = data.transpose(3, 1, 2, 0)
    return data


def preprocess_one_sample(data_numpy, config):
    """
    Mimics the Feeder.__getitem__ processing for a single sample.
    data_numpy: (C, T, V, M)
    """
    # 1. Normalize/Centering (Crucial for accuracy)
    # Subtract the SpineBase (joint 20, index 0-based is 20 if using index 1-25)
    # Note: In NTU, 1-based index 21 is SpineBase. 0-based index 20.

    # Store trajectory before subtraction (Feeder logic)
    trajectory = data_numpy[:, :, 20, :]

    # Subtract spine base from all joints
    data_numpy = data_numpy - data_numpy[:, :, 20:21, :]

    # Restore trajectory at the spine base slot (Feeder logic)
    data_numpy[:, :, 20, :] = trajectory

    # 2. Modality Generation (Bone/Vel)
    modality = 'joint'
    if config['train_feeder_args'].get('bone', False):
        modality = 'bone'
    elif config['train_feeder_args'].get('vel', False):
        modality = 'vel'

    joint = data_numpy.copy()  # Raw coordinates for embedding/graph

    if modality == 'bone':
        from feeders.bone_pairs import ntu_pairs
        bone_data = np.zeros_like(data_numpy)
        for v1, v2 in ntu_pairs:
            # v1-1, v2-1 because pairs are usually 1-indexed
            bone_data[:, :, v1 - 1, :] = data_numpy[:, :, v1 - 1, :] - data_numpy[:, :, v2 - 1, :]

        # Keep spine center trajectory
        bone_data[:, :, 20, :] = data_numpy[:, :, 20, :]
        data_numpy = bone_data

    elif modality == 'vel':
        # Velocity: frame[t+1] - frame[t]
        # Pad last frame with 0
        data_numpy[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]
        data_numpy[:, -1, :, :] = 0

    return data_numpy, joint


def preprocess_window(data, joint, target_size=64):
    C, T, V, M = data.shape
    if T < target_size:
        pad_t = target_size - T
        data = np.pad(data, ((0, 0), (0, pad_t), (0, 0), (0, 0)), mode='constant')
        joint = np.pad(joint, ((0, 0), (0, pad_t), (0, 0), (0, 0)), mode='constant')

    # If larger, we only take the first chunk here for safety,
    # but sliding window handles the rest.
    return data[:, :target_size, :, :], joint[:, :target_size, :, :]


def sliding_window_inference(model, data, joint, window_size, stride, device):
    model.eval()
    C, T, V, M = data.shape

    if T <= window_size:
        windows_data = [data]
        windows_joint = [joint]
    else:
        windows_data = []
        windows_joint = []
        for t_start in range(0, T - window_size + 1, stride):
            t_end = t_start + window_size
            windows_data.append(data[:, t_start:t_end, :, :])
            windows_joint.append(joint[:, t_start:t_end, :, :])

        # Handle remainder
        if (T - window_size) % stride != 0:
            windows_data.append(data[:, -window_size:, :, :])
            windows_joint.append(joint[:, -window_size:, :, :])

    batch_scores = []
    with torch.no_grad():
        for w_d, w_j in zip(windows_data, windows_joint):
            w_d, w_j = preprocess_window(w_d, w_j, window_size)

            inp_d = torch.tensor(w_d, dtype=torch.float32).unsqueeze(0).to(device)
            inp_j = torch.tensor(w_j, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward
            output, _ = model(inp_d, None, inp_j)
            probs = F.softmax(output, dim=1)
            batch_scores.append(probs.cpu().numpy())

    if len(batch_scores) == 0:
        return np.zeros(60)  # Fallback

    final_score = np.mean(np.concatenate(batch_scores, axis=0), axis=0)
    return final_score


def get_parser():
    parser = argparse.ArgumentParser(description='Inference on Raw Skeleton File')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt model weights')

    # CHANGED: Now accepts a file path instead of an index
    parser.add_argument('--skeleton_path', required=True, help='Path to .skeleton file')

    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    return parser


def main():
    args = get_parser().parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 2. Load Model
    print(f"Initializing Model: {config['model']}")
    Model = import_class(config['model'])
    model = Model(**config['model_args'])
    model.to(args.device)

    # 3. Load Weights
    print(f"Loading weights: {args.checkpoint}")
    weights = torch.load(args.checkpoint)
    new_weights = OrderedDict()
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_weights[name] = v
    model.load_state_dict(new_weights, strict=False)

    # 4. Read Raw Skeleton
    print(f"Reading skeleton file: {args.skeleton_path}")
    raw_numpy = read_skeleton_file(args.skeleton_path)  # (C, T, V, M)

    # 5. Preprocess (Center, Generate Bone/Vel)
    # Note: We need to pass the full config or just the relevant args
    # constructing a minimal config dict if needed, or passing full loaded config
    data, joint = preprocess_one_sample(raw_numpy, config)

    # 6. Inference
    print("Running Inference...")
    scores = sliding_window_inference(model, data, joint, args.window_size, args.stride, args.device)

    pred_idx = np.argmax(scores)
    confidence = scores[pred_idx]

    print("\n" + "=" * 30)
    print("      RESULT      ")
    print("=" * 30)
    print(f"Input File      : {os.path.basename(args.skeleton_path)}")
    print(f"Predicted Class : {pred_idx}")
    print(f"Confidence      : {confidence:.4f}")
    print("=" * 30)


if __name__ == '__main__':
    main()