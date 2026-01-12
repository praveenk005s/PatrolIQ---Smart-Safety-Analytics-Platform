import mlflow

def get_best_clustering_run():
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("PatrolIQ_Crime_Clustering")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.silhouette_score DESC"],
        max_results=1
    )

    if not runs:
        return None

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "algorithm": run.data.params.get("algorithm"),
        "n_clusters": run.data.params.get("n_clusters"),
        "silhouette_score": run.data.metrics.get("silhouette_score")
    }


def get_pca_run():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("PatrolIQ_PCA_Analysis")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.explained_variance DESC"],
        max_results=1
    )

    if not runs:
        return None

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "n_components": run.data.params.get("n_components"),
        "explained_variance": run.data.metrics.get("explained_variance")
    }
