from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset
from tech_challenge_fase2.ga import GAConfig
from tech_challenge_fase2.optimization import run_experiment


def test_small_genetic_experiment_produces_valid_metrics():
    split = split_dataset(load_diagnostic_dataset())
    config = GAConfig(
        "teste",
        population_size=4,
        generations=1,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism=1,
        random_state=11,
    )
    result = run_experiment(
        "logistic_regression",
        split.X_train,
        split.y_train,
        config,
        cv_splits=3,
    )

    assert 0 <= result.best_fitness <= 1
    assert 0 <= result.cv_metrics["recall"] <= 1
    assert result.evaluations >= config.population_size

