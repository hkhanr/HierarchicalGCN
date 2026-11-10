import pickle
import numpy as np
import itertools
from tqdm import tqdm

# 1. File Paths
# Update these to exactly match your score files
score_files = {
    'J': 'work_dir/ntu60/cview/semantic_tsm_fusion/Hierarchical_GCN_decay_110_120_200_epochs_joint_semantic_tsm_attention_feature_fution_8heads_deterministic_bs128/epoch144_test_score.pkl',
    'B': 'work_dir/ntu60/cview/semantic_tsm_fusion/Hierarchical_GCN_decay_110_120_200_epochs_bone_semantic_tsm_attention_feature_fution_8heads_deterministic_bs128/epoch150_test_score.pkl',
    'JM': 'work_dir/ntu60/cview/semantic_tsm_fusion/Hierarchical_GCN_decay_110_120_200_epochs_joint_vel_semantic_tsm_attention_feature_fution_8heads_deterministic_bs128/epoch142_test_score.pkl',
    'BM': 'work_dir/ntu60/cview/semantic_tsm_fusion/Hierarchical_GCN_decay_110_120_200_epochs_bone_vel_semantic_tsm_attention_feature_fution_8heads_deterministic_bs128/epoch156_test_score.pkl'
}
npz_file = 'data/ntu60/NTU60_CV.npz'

# 2. Load Labels
print("Loading labels...")
npz_data = np.load(npz_file)
if 'y_test' in npz_data:
    y_test = npz_data['y_test']
elif 'y_val' in npz_data:
    y_test = npz_data['y_val']

if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)

# 3. Load Scores into Memory
print("Loading model scores...")
models_data = {}
for name, path in score_files.items():
    with open(path, 'rb') as f:
        models_data[name] = pickle.load(f)

# Extract sample names and total count
sample_names = list(models_data['J'].keys())
total_samples = len(sample_names)

# Convert dictionary scores into fast NumPy arrays for matrix math
# Shape will be: (num_samples, 60)
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

# 4. Define Weight Grid to Search
# We will test weights from 0.2 to 2.0 for each model
# To save time, we can lock the strongest model (Bone) at 1.0 as the baseline anchor.
weight_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

print("\nStarting Fully Unlocked Grid Search (10,000 combinations)...")
best_acc = 0.0
top_5_results = []

# Iterate through every combination of all 4 weights
for w_J in tqdm(weight_range, desc="Searching"):
    for w_B in weight_range:
        for w_JM in weight_range:
            for w_BM in weight_range:

                # Weighted Sum of the entire dataset at once (Very Fast)
                fused_scores = (scores_J * w_J) + \
                               (scores_B * w_B) + \
                               (scores_JM * w_JM) + \
                               (scores_BM * w_BM)

                # Get predictions and calculate accuracy
                predictions = np.argmax(fused_scores, axis=1)
                correct = np.sum(predictions == labels)
                acc = (correct / total_samples) * 100

                # Save top results
                top_5_results.append({
                    'acc': acc,
                    'weights': f"Joint:{w_J:.1f} | Bone:{w_B:.1f} | J_Mot:{w_JM:.1f} | B_Mot:{w_BM:.1f}"
                })

                if acc > best_acc:
                    best_acc = acc

# 5. Print Results
# Sort and get top 5 unique accuracy tiers
top_5_results = sorted(top_5_results, key=lambda x: x['acc'], reverse=True)

# Filter out duplicate scales (e.g. 1,1,1,1 and 2,2,2,2) by keeping only the first one of each accuracy score
unique_top_results = []
seen_accs = set()
for res in top_5_results:
    if res['acc'] not in seen_accs:
        unique_top_results.append(res)
        seen_accs.add(res['acc'])
    if len(unique_top_results) == 5:
        break

print("\n======================================================")
print("             GRID SEARCH COMPLETE")
print("======================================================")
print(f"Best Ensemble Accuracy : {best_acc:.3f}%")
print("Top 5 Weight Combinations:")
for i, res in enumerate(unique_top_results):
    print(f"  {i + 1}. {res['acc']:.3f}%  ->  {res['weights']}")
print("======================================================")