from tech_challenge_fase2.ga import GAConfig, GeneSpec, GeneticAlgorithm


def test_ga_is_reproducible_and_preserves_gene_limits():
    space = {
        "x": GeneSpec(kind="int", low=0, high=10),
        "y": GeneSpec(kind="float", low=0.0, high=1.0),
        "mode": GeneSpec(kind="choice", choices=("a", "b")),
    }

    def fitness(genes):
        score = genes["x"] + genes["y"] + (1 if genes["mode"] == "b" else 0)
        return score, {"score": score}

    config = GAConfig("teste", 8, 4, 0.8, 0.2, random_state=7)
    first = GeneticAlgorithm(space, fitness, config).run()
    second = GeneticAlgorithm(space, fitness, config).run()

    assert first.best_genes == second.best_genes
    assert first.best_fitness == second.best_fitness
    assert space["x"].contains(first.best_genes["x"])
    assert space["y"].contains(first.best_genes["y"])
    assert space["mode"].contains(first.best_genes["mode"])
    assert len(first.history) == config.generations + 1

