"""
run_all.py -- Chay toan bo pipeline Task 1 -> 2 -> 3.

Usage:
    cd part2
    python run_all.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Prevent blocking on plt.show() in script mode

# -- Add part2 to path --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    LAMBDAS, RANDOM_STATE, N_SPLITS,
)
from data_pipeline import prepare_data, calculate_vif, select_features_by_vif
from model_comparison import (
    fit_ols_full, fit_ols_selected,
    fit_ridge_cv, fit_lasso_cv,
    evaluate_models, get_best_model,
    plot_cv_results,
    run_residual_diagnostics,
    plot_feature_importance,
)


def main():
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    csv_path = os.path.join(os.path.dirname(__file__), "..", "raw_ebay.csv")
    df = pd.read_csv(csv_path)
    print(f"[1] Raw data loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # =========================================================================
    # 2. PREPARE DATA (Task 2)
    # =========================================================================
    X_train, X_test, y_train, y_test, feature_names, pipeline = prepare_data(
        df, TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    print(f"\n[2] Data prepared:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_test:  {X_test.shape}")
    print(f"    Features: {feature_names}")
    print(f"    Missing train: {X_train.isnull().sum().sum()}")
    print(f"    Missing test:  {X_test.isnull().sum().sum()}")

    # =========================================================================
    # 3. VIF & FEATURE SELECTION (Task 2)
    # =========================================================================
    print(f"\n[3] VIF Analysis:")
    vif_df = calculate_vif(X_train)
    print(vif_df.to_string(index=False))

    selected_features, final_vif = select_features_by_vif(X_train, threshold=10.0)
    print(f"\n    Selected features (VIF < 10): {selected_features}")

    # =========================================================================
    # 4. FIT MODELS (Task 3)
    # =========================================================================
    print(f"\n[4] Fitting models...")

    # Model 1: OLS Full
    model_ols_full = fit_ols_full(X_train, y_train)
    print(f"    OLS Full: {len(feature_names)} features")

    # Model 2: OLS Selected
    model_ols_selected = fit_ols_selected(X_train, y_train, selected_features)
    print(f"    OLS Selected: {len(selected_features)} features")

    # Model 3: Ridge CV
    print(f"    Ridge CV: testing {len(LAMBDAS)} lambdas x {N_SPLITS}-fold...")
    model_ridge, best_lam_r, cv_ridge = fit_ridge_cv(X_train, y_train, LAMBDAS)
    print(f"    -> Best lambda = {best_lam_r:.6g}")

    # Model 4: Lasso CV
    print(f"    Lasso CV: testing {len(LAMBDAS)} lambdas x {N_SPLITS}-fold...")
    model_lasso, best_lam_l, cv_lasso = fit_lasso_cv(X_train, y_train, LAMBDAS)
    print(f"    -> Best lambda = {best_lam_l:.6g}")

    # =========================================================================
    # 5. EVALUATE ON TEST SET (Task 3)
    # =========================================================================
    print(f"\n[5] Evaluation on TEST SET:")

    # OLS Selected dung X_test chi voi selected features
    X_test_selected = X_test[selected_features]

    models = {
        "OLS Full": model_ols_full,
        "OLS Selected": model_ols_selected,
        "Ridge CV": model_ridge,
        "Lasso CV": model_lasso,
    }

    # X_test khac nhau cho tung model
    X_tests = {
        "OLS Full": X_test,
        "OLS Selected": X_test_selected,
        "Ridge CV": X_test,
        "Lasso CV": X_test,
    }

    results = evaluate_models(models, X_tests, y_test)
    print(results.to_string(index=False))

    best = get_best_model(results, "RMSE")
    print(f"\n    Best model (by RMSE): {best}")

    # =========================================================================
    # 6. PLOTS (Task 3)
    # =========================================================================
    print(f"\n[6] Generating plots...")

    # CV Results
    plot_cv_results(cv_ridge, title="Ridge -- CV RMSE vs Lambda")
    plot_cv_results(cv_lasso, title="Lasso -- CV RMSE vs Lambda")

    # Residual Diagnostics cho best model
    best_model = models[best]
    best_X_test = X_tests[best]
    best_features = selected_features if best == "OLS Selected" else feature_names

    run_residual_diagnostics(best_model, best_X_test, y_test, model_name=best)

    # Feature Importance
    plot_feature_importance(best_model, best_features, top_n=20,
                            title=f"Feature Importance -- {best}")

    print("\n[DONE] All tasks completed successfully!")


if __name__ == "__main__":
    main()
