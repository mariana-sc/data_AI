import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric


def add_remaining_useful_life(df):
    """
    Add the remaining useful life to the dataframe
    :param df: Dataframes from the CMAPSS dataset
    :type df: pd.DataFrame
    :return: Dataframe with the remaining useful life added
    :rtype: pd.DataFrame
    """
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


def create_dataset(df, pcs):
    """
    Create a dataset from the dataframe from the principal component analysis
    :param df: pca_df
    :type df: pd.DataFrame
    :param pcs: Principal components to use e.g. [8,9]
    :type pcs: list
    :return: Dataset for the clustering analysis
    :rtype: list
    """
    if 1 not in pcs or 2 not in pcs:
        raise ValueError('You picked the wrong components')
    return [np.array([pc1, pc2]) for pc1, pc2 in zip(df['Principal Component 1'], df['Principal Component 2'])]


def create_dataset_admin(df):
    """
    Create a dataset from the dataframe from the principal component analysis_admin use (components 1 and 2 preselected)
    :param df: pca_df
    :type df: pd.DataFrame
    """
    return [np.array([pc1, pc2]) for pc1, pc2 in zip(df['Principal Component 1'], df['Principal Component 2'])]


def create_test_df():
    """
    This function creates the test dataframe for the clustering analysis
    :return: test dataframe
    :rtype: pd.DataFrame
    """
    test = np.zeros((7, 2))
    test[:, 0] = [2, 2, 3, 3, 2, 0, 2]
    test[:, 1] = [2, 3, 2, 2, 1, 2, 0]

    test_df = pd.DataFrame(test, columns=[
        'Principal Component 1',
        'Principal Component 2',

    ])
    return test_df


def create_test_dataset():
    """
    Thin function creates the test dataset for the clustering analysis
    :return: test dataset
    :rtype: list of np.arrays
    """
    return create_dataset_admin(create_test_df())


def get_results(clusters, dataset):
    """
    Get the results from the clustering analysis in the correct format for silhouette_score and plot_results
    :param clusters: Results from the clustering analysis
    :type clusters: list
    :param dataset: Input dataset
    :type dataset: list
    :return: clusters in the correct format
    :rtype: list
    """
    labels = copy.deepcopy(dataset)
    for i in range(len(clusters)):
        for idx in range(len(dataset)):
            if idx in clusters[i]:
                labels[idx] = i
    return labels


def plot_results(pca_df, test_df, clusters_train, clusters_test, centers, metric):
    """
    Plot the results from the clustering analysis
    :param pca_df: Dataframe from the principal component analysis
    :type pca_df: pd.DataFrame
    :param test_df: Dataframe from the test set
    :type test_df: pd.DataFrame
    :param clusters_train: Results from the clustering analysis for the train set straight from the clustering analysis
    (DONT USE THE get_results FUNCTION)
    :type clusters_train: list
    :param clusters_test: Results from the clustering analysis for the test set straight from the clustering analysis
    (DONT USE THE get_results FUNCTION)
    :type clusters_test: list
    :param centers: Cluster centers
    :type centers: list
    :param metric: Metric used for the clustering analysis 'Euclidean' or 'Manhattan'
    :type metric: str
    :return: Plot of the results
    """
    # Prepare the data
    plt.figure(figsize=(10, 6))

    dataset = create_dataset_admin(pca_df)
    clusters_train = get_results(clusters_train, dataset)

    results_man_df = copy.deepcopy(pca_df)
    results_test = copy.deepcopy(test_df)

    results_man_df['clusters'] = clusters_train
    results_test['clusters'] = clusters_test

    # Plot train data
    plt.scatter(results_man_df['Principal Component 1'], results_man_df['Principal Component 2'], marker="s",
                c=results_man_df['clusters'], cmap='plasma', edgecolors='k', s=120, alpha=0.8, label='Train Data')

    # Plot test data
    plt.scatter(test_df['Principal Component 1'], test_df['Principal Component 2'], marker="^",
                c=results_test['clusters'], cmap='plasma', edgecolors='k', s=150, alpha=0.8, label='Test Data')

    # Plot cluster centers

    centers = np.array(centers)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

    plt.title(f"Kmeans Clustering after PCA_{metric} Distance")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(visible=False)
    plt.legend()
    plt.show()


class kmeans_mod(kmeans):
    """
    This class is a modified version of the kmeans class from the pyclustering library with the predict method debugged
    """

    def predict(self, points):
        """!
        @brief Calculates the closest cluster to each point.

        @param[in] points (array_like): Points for which closest clusters are calculated.

        @return (list) List of closest clusters for each point. Each cluster is denoted by index. Return empty
                  collection if 'process()' method was not called.

        """

        nppoints = np.array(points)
        if len(self._kmeans__clusters) == 0:
            return []

        differences = np.zeros((len(nppoints), len(self._kmeans__centers)))
        for index_point in range(len(nppoints)):
            if self._kmeans__metric.get_type() != type_metric.USER_DEFINED:
                differences[index_point] = self._kmeans__metric(nppoints[index_point], np.array(self._kmeans__centers))
            else:
                differences[index_point] = [self._kmeans__metric(nppoints[index_point], center) for center in
                                            np.array(self._kmeans__centers)]

        return np.argmin(differences, axis=1)
