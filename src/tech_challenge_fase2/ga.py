"""Algoritmo genético independente de bibliotecas de otimização."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any, Callable, Literal


GeneKind = Literal["int", "float", "choice"]
Genome = dict[str, Any]
FitnessFunction = Callable[[Genome], tuple[float, dict[str, float]]]


@dataclass(frozen=True)
class GeneSpec:
    kind: GeneKind
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("Gene categórico sem opções.")
            return rng.choice(self.choices)
        if self.low is None or self.high is None:
            raise ValueError("Gene numérico sem limites.")
        if self.kind == "int":
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(float(self.low), float(self.high))

    def contains(self, value: Any) -> bool:
        if self.kind == "choice":
            return value in self.choices
        if self.low is None or self.high is None:
            return False
        return float(self.low) <= float(value) <= float(self.high)


@dataclass(frozen=True)
class GAConfig:
    name: str
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int = 3
    elitism: int = 2
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("A população deve ter ao menos dois indivíduos.")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("Taxas de crossover e mutação devem estar entre 0 e 1.")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("Elitismo inválido para o tamanho da população.")


@dataclass
class Individual:
    genes: Genome
    fitness: float = -math.inf
    details: dict[str, float] = field(default_factory=dict)

    def clone(self) -> "Individual":
        return Individual(dict(self.genes), self.fitness, dict(self.details))


@dataclass(frozen=True)
class GAResult:
    best_genes: Genome
    best_fitness: float
    best_details: dict[str, float]
    history: list[dict[str, float | int]]
    evaluations: int


class GeneticAlgorithm:
    """GA com seleção por torneio, crossover uniforme, mutação e elitismo."""

    def __init__(
        self,
        gene_space: dict[str, GeneSpec],
        fitness_function: FitnessFunction,
        config: GAConfig,
    ) -> None:
        if not gene_space:
            raise ValueError("O espaço genético não pode ser vazio.")
        self.gene_space = gene_space
        self.fitness_function = fitness_function
        self.config = config
        self.rng = random.Random(config.random_state)
        self._cache: dict[tuple[tuple[str, str], ...], tuple[float, dict[str, float]]] = {}

    @staticmethod
    def _key(genes: Genome) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((name, repr(value)) for name, value in genes.items()))

    def random_genome(self) -> Genome:
        return {name: spec.sample(self.rng) for name, spec in self.gene_space.items()}

    def _evaluate(self, individual: Individual) -> None:
        key = self._key(individual.genes)
        if key not in self._cache:
            fitness, details = self.fitness_function(dict(individual.genes))
            self._cache[key] = (float(fitness), dict(details))
        individual.fitness, individual.details = self._cache[key]

    def _tournament(self, population: list[Individual]) -> Individual:
        size = min(self.config.tournament_size, len(population))
        contenders = self.rng.sample(population, size)
        return max(contenders, key=lambda item: item.fitness)

    def crossover(self, left: Genome, right: Genome) -> tuple[Genome, Genome]:
        child_left, child_right = dict(left), dict(right)
        if self.rng.random() > self.config.crossover_rate:
            return child_left, child_right
        for name in self.gene_space:
            if self.rng.random() < 0.5:
                child_left[name], child_right[name] = child_right[name], child_left[name]
        return child_left, child_right

    def mutate(self, genes: Genome) -> Genome:
        mutated = dict(genes)
        for name, spec in self.gene_space.items():
            if self.rng.random() < self.config.mutation_rate:
                mutated[name] = spec.sample(self.rng)
        return mutated

    def _assert_valid(self, genes: Genome) -> None:
        for name, spec in self.gene_space.items():
            if name not in genes or not spec.contains(genes[name]):
                raise ValueError(f"Gene inválido: {name}={genes.get(name)!r}.")

    def run(self) -> GAResult:
        population = [Individual(self.random_genome()) for _ in range(self.config.population_size)]
        history: list[dict[str, float | int]] = []

        for generation in range(self.config.generations + 1):
            for individual in population:
                self._assert_valid(individual.genes)
                self._evaluate(individual)
            population.sort(key=lambda item: item.fitness, reverse=True)
            fitnesses = [item.fitness for item in population]
            history.append(
                {
                    "generation": generation,
                    "best": fitnesses[0],
                    "mean": fmean(fitnesses),
                    "std": pstdev(fitnesses),
                }
            )
            if generation == self.config.generations:
                break

            next_population = [item.clone() for item in population[: self.config.elitism]]
            while len(next_population) < self.config.population_size:
                parent_left = self._tournament(population)
                parent_right = self._tournament(population)
                genes_left, genes_right = self.crossover(parent_left.genes, parent_right.genes)
                for genes in (self.mutate(genes_left), self.mutate(genes_right)):
                    self._assert_valid(genes)
                    next_population.append(Individual(genes))
                    if len(next_population) == self.config.population_size:
                        break
            population = next_population

        best = max(population, key=lambda item: item.fitness)
        return GAResult(
            best_genes=dict(best.genes),
            best_fitness=float(best.fitness),
            best_details=dict(best.details),
            history=history,
            evaluations=len(self._cache),
        )

