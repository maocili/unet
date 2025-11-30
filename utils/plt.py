import matplotlib.pyplot as plt
import torch
import numpy as np


def show_loss_plt(data, title="Training Metrics"):
    if not data:
        print("Warning: data is empty.")
        return

    first_key = next(iter(data))
    epochs = range(1, len(data[first_key]) + 1)

    plt.figure(figsize=(10, 6))

    for key in sorted(data.keys()):
        plt.plot(epochs, data[key], label=key, linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def show_predictions(pairs_list):
    rows = len(pairs_list)
    if rows == 0:
        print("Warning: pairs_list is empty.")
        return

    cols = len(pairs_list[0])

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])  # (1, 1)
    elif rows == 1:
        axes = axes.reshape(1, cols)  # (1, cols)
    elif cols == 1:
        axes = axes.reshape(rows, 1)  # (rows, 1)

    for i in range(rows):
        row_data = pairs_list[i]

        if len(row_data) != cols:
            print(f"Warning: Row {i} has {len(row_data)} items, expected {cols}. Skipping.")
            continue

        for j in range(cols):
            img_data = row_data[j]

            if isinstance(img_data, torch.Tensor):
                img_data = img_data.cpu().detach().numpy()

            img_data = np.squeeze(img_data)
            ax = axes[i, j]
            ax.imshow(img_data, cmap='gray')
            if j == 0:
                title = "(Original Image)"
            else:
                title = f"(Prediction Image)"

            ax.set_title(title)
            ax.axis('off')

    fig.suptitle("Visualization of Pseudo-label Generation", fontsize=20)
    plt.tight_layout()
    plt.show()
