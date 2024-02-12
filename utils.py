import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def create_dataset(df):
    return [np.array([pc1, pc2]) for pc1, pc2 in zip(df['Principal Component 1'], df['Principal Component 2'])]


def get_results(clusters, dataset):
    labels=copy.deepcopy(dataset)
    for i in range(len(clusters)):
      for idx in range(len(dataset)):
        if idx in clusters[i]:
          labels[idx]=i
    return labels


def plot_results(pca_df, clusters, centers):

    # Prepare the data
    plt.figure(figsize=(10, 6))

    dataset = create_dataset(pca_df)
    clusters_res = get_results(clusters, dataset)

    results_man_df = copy.deepcopy(pca_df)
    results_man_df['clusters'] = clusters_res
    results_df_1 = results_man_df[results_man_df['target'] == 1]
    results_df_3 = results_man_df[results_man_df['target'] == 3]

    # Plot data points for each dataset
    plt.scatter(results_df_1['Principal Component 1'], results_df_1['Principal Component 2'], marker="s",
                c=results_df_1['clusters'], cmap='plasma', edgecolors='k', s=120, alpha=0.8, label='Dataset 001')

    plt.scatter(results_df_3['Principal Component 1'], results_df_3['Principal Component 2'], marker="*",
                c=results_df_3['clusters'], cmap='plasma', edgecolors='k', s=100, alpha=0.8, label='Dataset 003')

    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

    plt.title('K-means Clustering after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(visible=False)
    plt.legend()
    plt.show()


