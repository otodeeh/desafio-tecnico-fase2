from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset


def test_dataset_and_split_are_reproducible():
    dataset = load_diagnostic_dataset()
    first = split_dataset(dataset)
    second = split_dataset(dataset)

    assert dataset.features.shape == (569, 30)
    assert set(dataset.target.unique()) == {0, 1}
    assert first.X_test.index.tolist() == second.X_test.index.tolist()
    assert first.y_train.mean() == second.y_train.mean()

