import numpy as np
import pandas as pd

from src.evaluation.ab_testing import run_ab_test, two_proportion_z_test


def test_two_proportion_z_test_p_value_in_valid_range():
    _, p_value = two_proportion_z_test(30, 500, 45, 500)
    assert 0.0 <= p_value <= 1.0


def test_two_proportion_z_test_identical_groups_gives_p_value_one():
    z_stat, p_value = two_proportion_z_test(50, 500, 50, 500)
    assert z_stat == 0.0
    assert p_value == 1.0


def _sample_df(n=400, churn_prob=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "churn": rng.binomial(1, churn_prob, size=n),
    })


def test_run_ab_test_p_value_in_valid_range():
    result = run_ab_test(_sample_df(), random_state=0)
    assert 0.0 <= result["p_value"] <= 1.0


def test_run_ab_test_without_simulated_effect_p_value_in_valid_range():
    result = run_ab_test(_sample_df(), simulate_effect=False, random_state=1)
    assert 0.0 <= result["p_value"] <= 1.0
