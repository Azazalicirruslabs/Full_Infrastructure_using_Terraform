"""
Sensitive Feature Detector for BiasLens™ Analyzer
Ported from fairness service for automatic sensitive feature detection.

This module provides HSIC Lasso-based detection of sensitive features
that have statistical dependency on the target variable.
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyHSICLasso import HSICLasso

# Force single-threaded execution for Windows compatibility
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress warnings
warnings.filterwarnings("ignore")

# Default threshold percentile for sensitivity detection
DEFAULT_HSIC_THRESHOLD_PERCENTILE = 75


class SensitiveFeatureDetector:
    """Detect sensitive features using HSIC Lasso library"""

    def __init__(self, threshold_percentile: int = DEFAULT_HSIC_THRESHOLD_PERCENTILE):

        self.threshold_percentile = threshold_percentile
        self.feature_scores: Dict = {}
        self.threshold_value: float = 0.0
        self.hsic_lasso = None
        self.feature_names: List[str] = []
        self.test_results: Dict = {}

    def detect_sensitive_features(
        self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = None
    ) -> Tuple[List[str], Dict, float]:
        """
        Detect sensitive features using HSIC Lasso.

        Args:
            X: Feature dataframe (excluding target column)
            y: Target variable series
            n_features_to_select: Number of top features to select (optional)

        Returns:
            Tuple of (sensitive_features_list, feature_scores_dict, threshold_value)
        """
        self.feature_names = X.columns.tolist()
        self.feature_scores = {}

        try:
            # Encode categorical features
            X_encoded, _ = self._encode_categorical(X)
            y_encoded, _ = self._encode_categorical(pd.DataFrame({y.name or "target": y}))
            y_array = y_encoded.iloc[:, 0].values

            # Get sample size
            n_samples = X_encoded.shape[0]
            n_features = X_encoded.shape[1]

            # Determine number of features to select
            if n_features_to_select is None:
                n_features_to_select = min(10, n_features)
            else:
                n_features_to_select = min(n_features_to_select, n_features)

            # For small datasets (< 50 samples), use simplified approach
            if n_samples < 50:
                return self._simple_correlation_analysis(X_encoded, y_array, n_features_to_select)

            # Create HSIC Lasso instance
            self.hsic_lasso = HSICLasso()

            # Prepare input for HSIC
            X_input, y_input = self._preprocess_for_hsic(X_encoded, y_array)

            # Input with discretize option
            self.hsic_lasso.input(X_input, y_input)

            # Run HSIC Lasso for regression (works for both classification and regression)
            self.hsic_lasso.regression(n_jobs=1)

            # Get results
            selected_indices = self.hsic_lasso.get_index()
            scores = self.hsic_lasso.get_index_score()

            # Map back to feature names and create scores dictionary
            if selected_indices is not None and len(selected_indices) > 0:
                for idx, score in zip(selected_indices, scores):
                    if idx < len(self.feature_names):
                        feature_name = self.feature_names[int(idx)]
                        self.feature_scores[feature_name] = float(score)

            # Add non-selected features with zero score
            for feature in self.feature_names:
                if feature not in self.feature_scores:
                    self.feature_scores[feature] = 0.0

            # Calculate threshold based on percentile
            all_scores = list(self.feature_scores.values())
            self.threshold_value = float(np.percentile(all_scores, self.threshold_percentile))

            # Identify sensitive features (above threshold)
            sensitive_features = [
                feature for feature, score in self.feature_scores.items() if score > self.threshold_value
            ]

            return sensitive_features, self.feature_scores, self.threshold_value

        except Exception as e:
            # Fallback to correlation-based method
            X_encoded, _ = self._encode_categorical(X)
            y_encoded, _ = self._encode_categorical(pd.DataFrame({y.name or "target": y}))
            y_array = y_encoded.iloc[:, 0].values
            return self._simple_correlation_analysis(X_encoded, y_array, n_features_to_select or 10)

    def _simple_correlation_analysis(
        self, X: pd.DataFrame, y: np.ndarray, n_features_to_select: int
    ) -> Tuple[List[str], Dict, float]:
        """
        Fallback method using correlation for small datasets or errors.
        """
        self.feature_scores = {}
        self.test_results = {}

        for feature in self.feature_names:
            try:
                feature_data = X[feature].values.astype(float)

                # Run appropriate statistical test
                test_result = self._statistical_dependency_test(feature_data, y)

                # Use correlation/effect size as the sensitivity score
                self.feature_scores[feature] = test_result["correlation"]
                self.test_results[feature] = test_result

            except Exception as e:
                self.feature_scores[feature] = 0.0
                self.test_results[feature] = {
                    "test_type": "error",
                    "p_value": 1.0,
                    "statistic": None,
                    "correlation": 0.0,
                }

        # Calculate threshold
        all_scores = np.array(list(self.feature_scores.values()))
        self.threshold_value = float(np.percentile(all_scores, self.threshold_percentile))

        # Identify sensitive features
        sensitive_features = [feature for feature, score in self.feature_scores.items() if score > self.threshold_value]

        return sensitive_features, self.feature_scores, self.threshold_value

    @staticmethod
    def _encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Encode categorical features to numerical."""
        df_encoded = df.copy()
        encoding_map = {}

        for col in df_encoded.columns:
            if df_encoded[col].dtype == "object":
                unique_values = df_encoded[col].unique()
                encoding_map[col] = {val: idx for idx, val in enumerate(unique_values)}
                df_encoded[col] = df_encoded[col].map(encoding_map[col])
            # Handle NaN values
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce").fillna(0)

        return df_encoded, encoding_map

    def _detect_feature_type(self, feature: np.ndarray) -> str:
        """Detect if feature is numerical or categorical."""
        unique_ratio = len(np.unique(feature)) / len(feature)

        # If < 20% unique values, treat as categorical
        if unique_ratio < 0.2:
            return "categorical"
        return "numerical"

    def _pearson_correlation_test(self, feature: np.ndarray, target: np.ndarray) -> Dict:
        """Pearson correlation test for numerical-numerical."""
        from scipy import stats

        correlation, p_value = stats.pearsonr(feature, target)

        return {
            "test_type": "pearson_correlation",
            "p_value": float(p_value),
            "statistic": float(abs(correlation)),
            "correlation": float(abs(correlation)),
        }

    def _anova_test(self, groups_feature: np.ndarray, values_target: np.ndarray, reversed: bool = False) -> Dict:
        """ANOVA F-test for categorical-numerical or numerical-categorical."""
        from scipy import stats

        # Group the target values by the categorical feature
        unique_groups = np.unique(groups_feature)
        grouped_values = [values_target[groups_feature == group] for group in unique_groups]

        # Remove empty groups
        grouped_values = [g for g in grouped_values if len(g) > 0]

        if len(grouped_values) < 2:
            return {"test_type": "anova", "p_value": 1.0, "statistic": 0.0, "correlation": 0.0}

        f_statistic, p_value = stats.f_oneway(*grouped_values)

        # Convert F-statistic to effect size (eta)
        correlation = float(np.sqrt(f_statistic / (f_statistic + len(values_target) - len(grouped_values))))

        return {
            "test_type": "anova",
            "p_value": float(p_value),
            "statistic": float(f_statistic),
            "correlation": correlation,
        }

    def _chi_square_test(self, feature: np.ndarray, target: np.ndarray) -> Dict:
        """Chi-square test for categorical-categorical."""
        from scipy import stats

        # Create contingency table
        contingency = pd.crosstab(feature, target)

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Calculate Cramér's V (effect size for chi-square)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

        return {
            "test_type": "chi_square",
            "p_value": float(p_value),
            "statistic": float(chi2),
            "correlation": cramers_v,
        }

    def _statistical_dependency_test(self, feature: np.ndarray, target: np.ndarray) -> Dict:
        """
        Unified test that chooses appropriate statistical test based on data types.
        """
        # Clean data
        mask = ~(np.isnan(feature) | np.isnan(target))
        feature_clean = feature[mask]
        target_clean = target[mask]

        if len(feature_clean) < 2:
            return {"test_type": "insufficient_data", "p_value": 1.0, "statistic": None, "correlation": 0.0}

        # Detect types
        feature_type = self._detect_feature_type(feature_clean)
        target_type = self._detect_feature_type(target_clean)

        # Choose appropriate test
        if feature_type == "numerical" and target_type == "numerical":
            return self._pearson_correlation_test(feature_clean, target_clean)

        elif feature_type == "categorical" and target_type == "numerical":
            return self._anova_test(feature_clean, target_clean)

        elif feature_type == "categorical" and target_type == "categorical":
            return self._chi_square_test(feature_clean, target_clean)

        elif feature_type == "numerical" and target_type == "categorical":
            return self._anova_test(target_clean, feature_clean, reversed=True)

        else:
            return {"test_type": "unsupported", "p_value": 1.0, "statistic": None, "correlation": 0.0}

    def get_feature_sensitivity_scores(self) -> List[Dict]:
        """Get detailed sensitivity scores for all features."""
        scores_list = []

        all_scores = np.array(list(self.feature_scores.values()))

        for feature, score in self.feature_scores.items():
            is_sensitive = score > self.threshold_value
            percentile_rank = float(np.mean(all_scores <= score) * 100) if len(all_scores) > 0 else 0.0

            # Include test type if available
            test_type = (
                self.test_results.get(feature, {}).get("test_type", "unknown")
                if hasattr(self, "test_results")
                else "unknown"
            )

            scores_list.append(
                {
                    "feature": feature,
                    "sensitivity_score": float(score),
                    "is_sensitive": is_sensitive,
                    "percentile_rank": percentile_rank,
                    "test_type": test_type,
                }
            )

        # Sort by score descending
        scores_list.sort(key=lambda x: x["sensitivity_score"], reverse=True)
        return scores_list

    def _preprocess_for_hsic(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data specifically for HSIC Lasso compatibility.
        """
        # Keep as integers - NO standardization, NO normalization
        X_processed = X.astype(np.int32).values.copy()
        y_processed = y.astype(np.int32).copy()

        # Ensure minimum value is 0 or positive
        for i in range(X_processed.shape[1]):
            col_min = X_processed[:, i].min()
            if col_min < 0:
                X_processed[:, i] = X_processed[:, i] - col_min

        y_min = y_processed.min()
        if y_min < 0:
            y_processed = y_processed - y_min

        # Verify no NaN or inf
        if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
            X_processed[~np.isfinite(X_processed)] = 0

        if np.any(np.isnan(y_processed)) or np.any(np.isinf(y_processed)):
            y_processed[~np.isfinite(y_processed)] = 0

        return X_processed, y_processed
