# Letter Recognition Qt C++
This Qt app trains a 2-hidden-layer feed-forward neural network (softmax output) 
to classify letters A–Z from 16 numeric features. It lets user:

1. Load the dataset
2. Choose hidden‐layer activation
3. Set learning rate and max epochs
4. Enable shuffle and an early-stop target PGC
5. Train and test on the training/test splits
6. Classify a single pattern
7. Save / load trained weights
8. Export per-epoch logs and a confusion matrix
9. Default RNG seed is 123 for reproducibility.

Files
Input
Dataset text file.
Outputs (generates in the app folder)
training_log.csv – per-epoch: epoch,train_MSE,train_PGC,test_MSE,test_PGC
confusion_matrix.csv – confusion matrix for the best checkpoint
weights.txt – saved weights

Controls
Buttons and Fields
1. Read File: pick the dataset text file.
2. Activation: choose Choose Activation Function.
3. Initialise Network: sets the chosen activation and re-randomises weights (seed=123).
4. Shuffle each epoch: toggles random shuffling per epoch.
5. Train Network (max epochs) + spin box – number of training epochs.
6. Learning rate slider + LCD : shows the actual LR.
7. Test Network on TRAINING SET: runs evaluation over the training split; updates PGC LCD.
8. Test Network on TEST SET: runs evaluation over the held-out test split; updates PGC LCD.
9. Save Weights: saves current weights to the filename shown above the button.
10. Load Weights – loads weights from the given filename.
11. Classify Test Pattern – classifies a single input pattern from the text box beside it.
