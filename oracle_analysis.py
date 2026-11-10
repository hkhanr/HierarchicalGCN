import pickle
import numpy as np

# 1. Define your files
# Update these paths to point to your actual score files and label file
score_files = {
    'Joint': 'work_dir/ntu60/cview/Hierarchical_GCN_decay_110_120_170_epochs_new_8heads_deterministic/epoch158_test_score.pkl',
    'Bone': 'work_dir/ntu60/cview/Hierarchical_GCN_decay_110_120_200_epochs_bone_new_8heads_deterministic/epoch168_test_score.pkl',
    'Joint Motion': 'work_dir/ntu60/cview/Hierarchical_GCN_decay_110_120_200_epochs_joint_vel_new_8heads_deterministic/epoch138_test_score.pkl',
    'Bone Motion': 'work_dir/ntu60/cview/Hierarchical_GCN_decay_110_120_200_epochs_bone_vel_new_8heads_deterministic/epoch145_test_score.pkl'
}


npz_file = 'data/ntu60/NTU60_CV.npz'

# 2. Load Ground Truth Labels from .npz
print(f"Loading labels from {npz_file}...")
npz_data = np.load(npz_file)

# The test labels are usually stored under 'y_test' or 'y_val'
if 'y_test' in npz_data:
    y_test = npz_data['y_test']
elif 'y_val' in npz_data:
    y_test = npz_data['y_val']
else:
    print(f"Available keys in NPZ: {npz_data.files}")
    raise KeyError("Could not find the test labels key in the NPZ file.")

# Some codebases store labels as one-hot encoded arrays.
# If so, convert them back to standard integers (0 to 59)
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)

# 3. Load the Score Data
models_data = {}
for model_name, file_path in score_files.items():
    try:
        with open(file_path, 'rb') as f:
            models_data[model_name] = pickle.load(f)  # Format: {'test_0': array(60,), ...}
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}. Skipping.")

# Trackers
individual_correct = {name: 0 for name in models_data.keys()}
oracle_correct = 0
hard_failures = 0

# Grab the list of sample names from the first loaded model (e.g., 'test_0', 'test_1')
reference_model = list(models_data.keys())[0]
sample_names = list(models_data[reference_model].keys())
total_samples = len(sample_names)

print(f"Loaded {total_samples} test samples. Calculating Oracle...")

# 4. Analyze overlap video by video
for sample_name in sample_names:
    # sample_name is something like 'test_42'. We extract the integer '42'
    idx = int(sample_name.split('_')[1])

    # Get the true label from the NPZ array
    true_label = int(y_test[idx])

    got_it_right = []

    # Check each model's prediction
    for model_name, scores_dict in models_data.items():
        if sample_name in scores_dict:
            score_array = scores_dict[sample_name]
            predicted_label = np.argmax(score_array)

            if predicted_label == true_label:
                individual_correct[model_name] += 1
                got_it_right.append(model_name)

    # Oracle Logic: If at least one model got it right, the Oracle wins
    if len(got_it_right) > 0:
        oracle_correct += 1
    else:
        # If the list is empty, every single model failed
        hard_failures += 1

# 5. Print the Reality Check
print("\n======================================================")
print("             INDIVIDUAL MODEL ACCURACIES")
print("======================================================")
for model_name, correct in individual_correct.items():
    acc = (correct / total_samples) * 100
    print(f"{model_name:<15}: {acc:.2f}%")

print("\n======================================================")
print("             ENSEMBLE POTENTIAL (ORACLE)")
print("======================================================")
oracle_acc = (oracle_correct / total_samples) * 100
hard_fail_pct = (hard_failures / total_samples) * 100

print(f"Oracle Accuracy : {oracle_acc:.2f}%  <-- Your theoretical maximum ensemble score!")
print(f"Hard Failures   : {hard_fail_pct:.2f}%  <-- Videos where EVERY model failed completely.")
print("======================================================")