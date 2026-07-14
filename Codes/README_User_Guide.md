
# Letter Recognition NN — User Guide

## Steps
1. Click **Read File** to load `complete_data_set.txt`.
2. Click **Initialise Network**.
3. Set **Learning Rate** and **Max Epochs** in the GUI.
4. Click **Train Network (Max Epochs)**. During training, `training_log.csv` is updated with `epoch,train_sse,test_sse`.
   - Weights auto-save every 100 epochs to the path in *Save Weights As* (default `best_weights.txt`).
5. Click **Test All Patterns** (or **Test on Training Set**) to evaluate accuracy.
6. To create a confusion matrix for the held-out test set (last 4,000 patterns), use **Tools → Save Confusion Matrix** (or run once training completes).
   - The matrix is saved to `confusion_matrix.csv`.
7. You can load weights via **Load Weights** and classify a custom input pattern in the GUI.

## Generated Artifacts
- `best_weights.txt` — best/last saved weights.
- `training_log.csv` — per-epoch SSE (train & test).
- `confusion_matrix.csv` — confusion matrix (actual rows vs predicted columns).
