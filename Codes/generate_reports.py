
import pandas as pd
import matplotlib.pyplot as plt

def plot_mse(log_csv='training_log.csv', out_png='mse_vs_epochs.png'):
    df = pd.read_csv(log_csv)
    plt.figure()
    plt.plot(df['epoch'], df['train_sse'], label='Train SSE')
    plt.plot(df['epoch'], df['test_sse'], label='Test SSE')
    plt.xlabel('Epoch')
    plt.ylabel('SSE')
    plt.title('MSE (SSE) vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")

def plot_confusion(cm_csv='confusion_matrix.csv', out_png='confusion_matrix.png'):
    import numpy as np
    import csv
    rows = []
    with open(cm_csv, 'r') as f:
        r = csv.reader(f)
        header = next(r)  # skip header
        for row in r:
            rows.append([int(x) for x in row[1:]])
    mat = np.array(rows)
    plt.figure()
    plt.imshow(mat, aspect='auto')
    plt.colorbar()
    plt.xlabel('Predicted class index')
    plt.ylabel('Actual class index')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")

if __name__ == '__main__':
    plot_mse()
    # Only plots if the CSV is present; ignore errors
    try:
        plot_confusion()
    except Exception as e:
        print("Confusion plotting skipped:", e)
