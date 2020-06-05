from functools import partial
from random import seed

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import ComposerRequirements
from core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposer
from core.composer.node import preprocessing_for_tasks
from core.composer.random_composer import RandomSearchComposer, History
from core.models.data import InputData
from core.models.data import train_test_data_setup
from core.models.preprocessing import Normalization
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository, \
    ComplexityMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (chain_template_balanced_tree, fit_template,
                                        show_chain_template, real_chain)
from experiments.composer_benchmark import to_labels
from experiments.exp_generate_data import synthetic_dataset
from experiments.tree_dist import chain_distance
from experiments.viz import show_history_optimization_comparison, show_tree_distance_changes

seed(42)
np.random.seed(42)


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.dt]
    return models


def source_chain(model_types, samples, features, classes):
    template = chain_template_balanced_tree(model_types=model_types, depth=2, models_per_level=[2, 1],
                                            samples=samples, features=features)
    show_chain_template(template)
    fit_template(template, classes=classes, with_gaussian=True, skip_fit=False)
    initialized_chain = real_chain(template)

    return initialized_chain


def data_generated_by(chain, samples, features_amount, classes):
    task_type = MachineLearningTasksEnum.classification
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    train = InputData(idx=np.arange(0, samples),
                      features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=train).predict
    synth_labels = to_labels(synth_target)
    data_synth_train = InputData(idx=np.arange(0, samples),
                                 features=features, target=synth_labels, task_type=task_type)

    # data_synth_train.features = Normalization().fit(data_synth_train.features).apply(data_synth_train.features)
    # preproc_data = copy(data_synth_train)
    # preprocessor = Normalization().fit(preproc_data.features)
    # preproc_data.features = preprocessor.apply(preproc_data.features)

    preprocessing_for_tasks[MachineLearningTasksEnum.classification] = Normalization

    chain.fit_from_scratch(input_data=data_synth_train)

    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    test = InputData(idx=np.arange(0, samples),
                     features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=test).predict
    synth_labels = to_labels(synth_target)
    data_synth_test = InputData(idx=np.arange(0, samples),
                                features=features, target=synth_labels, task_type=task_type)
    return data_synth_test


def _reduced_history_best(history_all, generations, pop_size):
    reduced_fitness = []
    reduced_distances = []
    history_, source_chain, = history_all
    for gen in range(generations):
        fitness_values, chains = [], []
        for individ in history_[gen * pop_size: (gen + 1) * pop_size]:
            chains.append(individ[0])
            fitness_values.append(abs(individ[1]))
        best = min(fitness_values)
        best_chain = chains[fitness_values.index(best)]
        print(f'Min in generation #{gen}: {best}')
        reduced_fitness.append(best)
        reduced_distances.append(chain_distance(source_chain, best_chain))
    return reduced_fitness, reduced_distances


def roc_score(chain, data_to_compose, data_to_validate):
    predicted_train = chain.predict(data_to_compose)
    predicted_test = chain.predict(data_to_validate)
    # the quality assessment for the simulation results
    roc_train = roc_auc(y_true=data_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')

    return roc_train, roc_test


def source_chain_self_predict():
    models_in_source_chain = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 10000, 10, 2

    train_score, test_score = [], []

    for _ in range(5):
        source = source_chain(models_in_source_chain, samples=samples,
                              features=features_amount, classes=classes)
        data_full = data_generated_by(source, samples, features_amount, classes)
        data_to_compose, data_to_validate = train_test_data_setup(data_full)
        source.fit_from_scratch(input_data=data_to_compose)
        roc_train, roc_test = roc_score(chain=source, data_to_compose=data_to_compose,
                                        data_to_validate=data_to_validate)
        train_score.append(roc_train)
        test_score.append(roc_test)
    print(train_score)
    print(test_score)


def _distances_history(source_chain, chain_history):
    distances = [chain_distance(source_chain, chain) for chain in chain_history]
    return distances


def compare_composers():
    runs = 1
    iterations = 5
    pop_size = 5
    models_in_source_chain = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 10000, 10, 2
    default_metric = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    distance_metric = MetricsRepository().metric_by_id(ComplexityMetricsEnum.chain_distance)

    history_random, history_gp = [], []
    random_dist, gp_dist = [], []
    for run in range(runs):
        source = source_chain(models_in_source_chain, samples=samples,
                              features=features_amount, classes=classes)
        data_full = data_generated_by(source, samples, features_amount, classes)
        data_to_compose, data_to_validate = train_test_data_setup(data_full)
        available_model_types = models_to_use()
        metric = partial(distance_metric, source_chain=source)
        # Init and run RandomComposer
        print('Running RandomComposer:')
        random_composer = RandomSearchComposer(iter_num=iterations * pop_size)
        random_reqs = ComposerRequirements(primary=available_model_types, secondary=available_model_types)
        history_best_random = History()
        random_composed = random_composer.compose_chain(data=data_to_compose,
                                                        initial_chain=None,
                                                        composer_requirements=random_reqs,
                                                        metrics=metric,
                                                        history_callback=history_best_random)
        history_random.append(history_best_random.fitness_values)
        random_composed.fit(input_data=data_to_compose, verbose=True)
        roc_score(random_composed, data_to_compose, data_to_validate)
        random_dist.append(_distances_history(source, history_best_random.best_chains))
        # Init and run GPComposer
        print('Running GPComposer:')
        gp_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=2,
            max_depth=4, pop_size=pop_size, num_of_generations=iterations,
            crossover_prob=0.8, mutation_prob=0.4)
        gp_composer = GPComposer()
        gp_composed = gp_composer.compose_chain(data=data_to_compose,
                                                initial_chain=None,
                                                composer_requirements=gp_requirements,
                                                metrics=metric, is_visualise=False)
        history_gp.append((gp_composer.history, source))
        gp_composed.fit(input_data=data_to_compose, verbose=True)
        roc_score(gp_composed, data_to_compose, data_to_validate)
        gp_dist.append(chain_distance(source, gp_composed))

    reduced_fitness_gp = []
    reduced_distances_gp = []
    for history in history_gp:
        fitness, chains = _reduced_history_best(history, iterations, pop_size)
        reduced_fitness_gp.append(fitness)
        reduced_distances_gp.append(chains)

    show_history_optimization_comparison(first=history_random, second=reduced_fitness_gp,
                                         iterations_first=range(iterations * pop_size),
                                         iterations_second=[_ * pop_size for _ in range(iterations)],
                                         label_first='Random', label_second='GP')
    show_tree_distance_changes(random_dist, reduced_distances_gp, range(iterations * pop_size),
                               [_ * pop_size for _ in range(iterations)], label_first='Random', label_second='GP')


if __name__ == '__main__':
    compare_composers()
