"""A/B test simulation: does a retention campaign reduce churn?

Hypothesis: assigning at-risk customers to a retention campaign (treatment)
lowers their churn rate compared to customers who receive no intervention
(control).

No real campaign was ever run, so this module randomly splits customers
into treatment/control and *simulates* a campaign effect by re-rolling the
treatment group's churn outcome at a reduced probability. This exists to
demonstrate the statistical testing methodology (two-proportion z-test) end
to end, not to report a real historical result.
"""

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "data/processed/customers_clean.csv"
RANDOM_STATE = 42
TREATMENT_FRACTION = 0.5
SIMULATED_RELATIVE_EFFECT = -0.30  # campaign cuts churn probability by 30%
SIGNIFICANCE_LEVEL = 0.05


def assign_groups(df: pd.DataFrame, treatment_fraction: float = TREATMENT_FRACTION,
                   random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Randomly assign each customer to 'treatment' or 'control'."""
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df["group"] = rng.choice(
        ["treatment", "control"], size=len(df),
        p=[treatment_fraction, 1 - treatment_fraction],
    )
    return df


def simulate_campaign_effect(df: pd.DataFrame, relative_effect: float = SIMULATED_RELATIVE_EFFECT,
                              random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Re-roll churn for the treatment group at a reduced probability,
    simulating what a successful retention campaign might look like.
    Control group churn is left untouched (the real observed outcomes)."""
    rng = np.random.default_rng(random_state + 1)
    df = df.copy()
    baseline_rate = df.loc[df["group"] == "control", "churn"].mean()
    simulated_rate = max(0.0, baseline_rate * (1 + relative_effect))

    treatment_mask = df["group"] == "treatment"
    df.loc[treatment_mask, "churn"] = rng.binomial(1, simulated_rate, size=treatment_mask.sum())
    return df


def two_proportion_z_test(successes_a: int, n_a: int, successes_b: int, n_b: int) -> tuple:
    """Two-sided two-proportion z-test. Returns (z_statistic, p_value)."""
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    z_stat = 0.0 if se == 0 else (p_a - p_b) / se
    p_value = 2 * stats.norm.sf(abs(z_stat))
    return z_stat, p_value


def run_ab_test(df: pd.DataFrame, simulate_effect: bool = True,
                 treatment_fraction: float = TREATMENT_FRACTION,
                 random_state: int = RANDOM_STATE) -> dict:
    """Split into treatment/control, (optionally) simulate a campaign
    effect, and run a two-proportion z-test comparing churn rates."""
    df = assign_groups(df, treatment_fraction, random_state)
    if simulate_effect:
        df = simulate_campaign_effect(df, random_state=random_state)

    treatment = df[df["group"] == "treatment"]
    control = df[df["group"] == "control"]

    treatment_rate = float(treatment["churn"].mean())
    control_rate = float(control["churn"].mean())
    effect_size = treatment_rate - control_rate  # percentage points, negative = improvement

    z_stat, p_value = two_proportion_z_test(
        int(treatment["churn"].sum()), len(treatment),
        int(control["churn"].sum()), len(control),
    )

    return {
        "treatment_n": len(treatment),
        "control_n": len(control),
        "treatment_churn_rate": treatment_rate,
        "control_churn_rate": control_rate,
        "effect_size_pp": effect_size,
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "significant": p_value < SIGNIFICANCE_LEVEL,
    }


def print_conclusion(result: dict):
    direction = "reduced" if result["effect_size_pp"] < 0 else "increased"
    pp_change = abs(result["effect_size_pp"]) * 100
    relative_change = (
        abs(result["effect_size_pp"]) / result["control_churn_rate"] * 100
        if result["control_churn_rate"] > 0 else float("nan")
    )
    significance = (
        f"statistically significant at p < {SIGNIFICANCE_LEVEL}"
        if result["significant"]
        else f"not statistically significant at p < {SIGNIFICANCE_LEVEL}"
    )

    print("\nHypothesis: the retention campaign reduces churn rate vs. control.")
    print(f"Treatment group (n={result['treatment_n']}): churn rate = {result['treatment_churn_rate']:.2%}")
    print(f"Control group   (n={result['control_n']}): churn rate = {result['control_churn_rate']:.2%}")
    print(f"z-statistic = {result['z_statistic']:.3f}, p-value = {result['p_value']:.4f}")
    print(
        f"\nConclusion: the campaign {direction} churn by {pp_change:.2f} percentage points "
        f"({relative_change:.1f}% relative change), which is {significance} "
        f"(p = {result['p_value']:.4f})."
    )


def main():
    df = pd.read_csv(DATA_PATH)
    result = run_ab_test(df)
    print_conclusion(result)
    return result


if __name__ == "__main__":
    main()
