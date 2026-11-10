# HierarchicalGCN

Code for **HierarchicalGCN**, a hierarchical part-based graph convolutional network for skeleton-based action recognition.

The skeleton is split into six body parts (left/right arms, left/right legs, torso, head). Local encoders extract features inside each part; a compact 6-node supergraph then models cross-limb interaction with three lightweight modules:

- **ASR** — Adaptive Spatial Routing on the supergraph
- **TSM** — Temporal Shift Module
- **ECA** — Efficient Channel Attention

This repository is built on the [BlockGCN](https://github.com/ZhouYuxuanYX/BlockGCN) training codebase.

## Model

| File | Role |
|------|------|
| `model/HierarchicalGCN.py` | Paper model (ASR + TSM + ECA) |
| `model/BlockGCN.py` | Flat BlockGCN baseline (optional) |
| `graph/ntu_rgb_d.py` | NTU graph with part indices and supergraph |

Default configs under `config/` point at `model.HierarchicalGCN` with all three flags set to `True`.

```yaml
model: model.HierarchicalGCN
model_args:
  num_class: 60
  num_point: 25
  num_person: 2
  graph: graph.ntu_rgb_d.Graph
  graph_args:
    labeling_mode: 'spatial'
  use_asr: True
  use_tsm: True
  use_eca: True
```

## Preparation

### Environment

```bash
pip install -r requirements.txt
pip install -e torchlight
```

PyTorch with CUDA is required for training. The code was developed with PyTorch 1.13+.

`torch-topological` is listed as optional and is needed only if you run `model.BlockGCN.Model`.

### Download datasets

Supported datasets:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request the dataset: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only archives:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
3. Extract them to `./data/nturgbd_raw`

#### NW-UCLA

1. Download from [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data layout

```
- data/
  - NW-UCLA/
    - all_sqe
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/      # from nturgbd_skeletons_s001_to_s017.zip
    - nturgb+d_skeletons120/  # from nturgbd_skeletons_s018_to_s032.zip
```

Configs for NTU-60 expect processed files such as `data/ntu60/NTU60_CS.npz` (see `config/nturgbd-*/`). After generating data, place or symlink the `.npz` files to match those paths.

### Generate NTU data

```bash
cd ./data/ntu   # or cd ./data/ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

## Training and testing

### Training

```bash
bash train.sh
```

Or call `main.py` directly:

```bash
python main.py \
  --config config/nturgbd-cross-subject/default.yaml \
  --model model.HierarchicalGCN \
  --work-dir work_dir/ntu60/csub/HierarchicalGCN_joint \
  --device 0
```

Modality configs (joint / bone / joint motion / bone motion):

- `config/nturgbd-cross-subject/{default,bone,vel,bone_vel}.yaml`
- `config/nturgbd-cross-view/{default,bone,vel,bone_vel}.yaml`
- `config/nturgbd120-cross-subject/...`
- `config/nturgbd120-cross-set/...`

### Testing

```bash
bash evaluate.sh
```

Example:

```bash
python main.py \
  --weights path/to/checkpoint.pt \
  --phase test \
  --save-score True \
  --config config/nturgbd-cross-view/bone.yaml \
  --model model.HierarchicalGCN \
  --work-dir work_dir/ntu60/cview/bone \
  --device 0
```

### Four-stream ensemble

```bash
bash ensemble.sh
```

`ensemble.py` / `grid_search.py` fuse Joint, Bone, Joint Motion, and Bone Motion softmax scores.

## Acknowledgements

This repo builds on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [BlockGCN](https://github.com/ZhouYuxuanYX/BlockGCN). Data processing follows [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch). Training strategy follows [Hyperformer](https://github.com/ZhouYuxuanYX/Hyperformer).
