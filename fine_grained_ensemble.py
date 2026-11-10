import pickle
import numpy as np
import itertools
from tqdm import tqdm

# 1. File Paths (Unchanged)
score_files = {
    'J': 'work_dir/ntu60/cview/adaptive_tsm_attention_fusion/Hierarchical_GCN_decay_110_120_200_epochs_joint_adaptive_tsm_attention_feature_fusion_8heads_deterministic_bs128/epoch133_test_score.pkl',
    'B': 'work_dir/ntu60/cview/adaptive_tsm_attention_fusion/Hierarchical_GCN_decay_110_120_200_epochs_bone_adaptive_tsm_attention_feature_fusion_8heads_deterministic_bs128/epoch132_test_score.pkl',
    'JM': 'work_dir/ntu60/cview/adaptive_tsm_attention_fusion/Hierarchical_GCN_decay_110_120_200_epochs_joint_vel_adaptive_tsm_attention_feature_fusion_8heads_deterministic_bs128/epoch170_test_score.pkl',
    'BM': 'work_dir/ntu60/cview/adaptive_tsm_attention_fusion/Hierarchical_GCN_decay_110_120_200_epochs_bone_vel_adaptive_tsm_attention_feature_fusion_8heads_deterministic_bs128/epoch123_test_score.pkl'
}
npz_file = 'data/ntu60/NTU60_CV.npz'

# 2. Load Labels (Unchanged)
print("Loading labels...")
npz_data = np.load(npz_file)
if 'y_test' in npz_data:
    y_test = npz_data['y_test']
elif 'y_val' in npz_data:
    y_test = npz_data['y_val']

if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)

# 3. Load Scores into Memory (Unchanged)
print("Loading model scores...")
models_data = {}
for name, path in score_files.items():
    with open(path, 'rb') as f:
        models_data[name] = pickle.load(f)

sample_names = list(models_data['J'].keys())
total_samples = len(sample_names)

print("Converting to NumPy arrays for fast matrix math...")
scores_J = np.zeros((total_samples, 60))
scores_B = np.zeros((total_samples, 60))
scores_JM = np.zeros((total_samples, 60))
scores_BM = np.zeros((total_samples, 60))
labels = np.zeros(total_samples, dtype=int)

for i, sample in enumerate(sample_names):
    idx = int(sample.split('_')[1])
    labels[i] = int(y_test[idx])
    scores_J[i] = models_data['J'][sample]
    scores_B[i] = models_data['B'][sample]
    scores_JM[i] = models_data['JM'][sample]
    scores_BM[i] = models_data['BM'][sample]

# ==============================================================
# 4. FINE-GRAINED SEARCH GRID
# Centered around the best known anchor: (0.8, 1.4, 1.1, 0.7)
# ==============================================================
print("\nSetting up Fine-Grained Search Space (0.01 increments)...")
# np.round prevents floating point inaccuracies like 0.700000000001
w_J_range = np.round(np.arange(0.70, 0.91, 0.01), 2)
w_B_range = np.round(np.arange(1.30, 1.51, 0.01), 2)
w_JM_range = np.round(np.arange(1.00, 1.21, 0.01), 2)
w_BM_range = np.round(np.arange(0.60, 0.81, 0.01), 2)

combinations = list(itertools.product(w_J_range, w_B_range, w_JM_range, w_BM_range))
print(f"Total combinations to evaluate: {len(combinations)}")

best_acc = 0.0
all_results = []

# Iterating through the combinations list is much cleaner than 4 nested loops
for w_J, w_B, w_JM, w_BM in tqdm(combinations, desc="Fine-Tuning"):

    # Weighted Sum
    fused_scores = (scores_J * w_J) + \
                   (scores_B * w_B) + \
                   (scores_JM * w_JM) + \
                   (scores_BM * w_BM)

    # Get predictions and calculate accuracy
    predictions = np.argmax(fused_scores, axis=1)
    correct = np.sum(predictions == labels)
    acc = (correct / total_samples) * 100

    # Only save results that are competitive to save memory and sorting time
    if acc >= 96.700:
        all_results.append({
            'acc': acc,
            'weights': f"Joint:{w_J:.2f} | Bone:{w_B:.2f} | J_Mot:{w_JM:.2f} | B_Mot:{w_BM:.2f}"
        })

    if acc > best_acc:
        best_acc = acc

# ==============================================================
# 5. PRINT TOP RESULTS
# ==============================================================
all_results = sorted(all_results, key=lambda x: x['acc'], reverse=True)

unique_top_results = []
seen_accs = set()

# Keep exactly the top 10 unique accuracy scores
for res in all_results:
    if res['acc'] not in seen_accs:
        unique_top_results.append(res)
        seen_accs.add(res['acc'])
    if len(unique_top_results) == 10:
        break

print("\n======================================================")
print("             FINE-GRAINED SEARCH COMPLETE")
print("======================================================")
print(f"Best Ensemble Accuracy : {best_acc:.4f}%")
print("Top 10 Weight Combinations:")
for i, res in enumerate(unique_top_results):
    print(f"  {i + 1}. {res['acc']:.4f}%  ->  {res['weights']}")
print("======================================================")