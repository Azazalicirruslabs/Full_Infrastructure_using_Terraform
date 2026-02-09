"""
Performance Summary Core Logic

This module provides core business logic for performance summary analysis,
including model type detection, threshold evaluation, and LLM-based interpretation
using Claude via AWS Bedrock. Supports both Regression and Classification models.
"""

import io
import ipaddress
import json
import logging
import os
import pickle
import socket
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from services.mainflow.app.schemas.performance_schema import (
    ClassificationMetricsOutput,
    ClassificationPerformanceMetrics,
    MetricDetail,
    PerformanceInterpretation,
    PerformanceMetadata,
    PerformanceSummaryRequest,
    PerformanceSummaryResponse,
    RegressionMetricsOutput,
    RegressionPerformanceMetrics,
)
from services.mainflow.app.utils.helper_function import (
    get_default_allowed_host_suffixes,
    get_default_allowed_hosts,
    make_safe_request,
    validate_and_resolve_url,
)

logger = logging.getLogger(__name__)

# Try to import onnxruntime for ONNX model support
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed. ONNX model support will be unavailable.")

# Security: Allowed modules for safe model deserialization
SAFE_MODULES = {
    "sklearn",
    "numpy",
    "pandas",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "joblib",
    "builtins",
    "__builtin__",
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows loading of trusted modules.
    This prevents arbitrary code execution during deserialization.
    """

    def find_class(self, module, name):
        """Only allow safe modules to be loaded."""
        module_root = module.split(".")[0]

        if module_root not in SAFE_MODULES:
            raise pickle.UnpicklingError(
                f"Loading module '{module}' is not allowed for security reasons. "
                f"Only whitelisted ML modules are permitted."
            )

        return super().find_class(module, name)


class ModelWrapper:
    """Wrapper class for different model types (sklearn, ONNX)."""

    def __init__(self, model, model_type: str = "sklearn"):
        self.model = model
        self.model_type = model_type  # "sklearn" or "onnx"
        self._input_name = None

        # For ONNX models, cache the input name
        if model_type == "onnx" and ONNX_AVAILABLE:
            self._input_name = model.get_inputs()[0].name

    def predict(self, X):
        """Make predictions."""
        if self.model_type == "onnx":
            if isinstance(X, pd.DataFrame):
                X_array = X.values.astype(np.float32)
            else:
                X_array = np.array(X).astype(np.float32)
            result = self.model.run(None, {self._input_name: X_array})
            return result[0]
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities (if applicable)."""
        if self.model_type == "onnx":
            if isinstance(X, pd.DataFrame):
                X_array = X.values.astype(np.float32)
            else:
                X_array = np.array(X).astype(np.float32)
            result = self.model.run(None, {self._input_name: X_array})
            # ONNX classifiers typically return [labels, probabilities]
            if len(result) > 1:
                return result[1]
            raise AttributeError("ONNX model does not support predict_proba")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Model does not support predict_proba")


class PerformanceSummaryAnalyzer:
    """
    Performance summary analyzer with model detection and LLM interpretation.

    Args:
        bedrock_runtime: Optional pre-configured Bedrock client for testing
    """

    def __init__(self, bedrock_runtime=None):
        if bedrock_runtime is not None:
            self.bedrock_runtime = bedrock_runtime
        else:
            self.bedrock_runtime = None
            self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            import boto3

            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("REGION_LLM", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_LLM"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_LLM"),
            )
            logger.info("AWS Bedrock initialized successfully")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Bedrock initialization failed: %s. LLM analysis will be unavailable.", exc)
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str) -> str:
        """
        Invoke Claude via AWS Bedrock.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            Claude's response text
        """
        if not self.bedrock_runtime:
            return "LLM analysis unavailable: Bedrock not initialized"

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }

            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(request_body),
            )

            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["content"][0]["text"]
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Claude invocation failed: %s", exc)
            return f"LLM analysis failed: {str(exc)}"

    def _validate_model_source(self, path: str) -> None:
        """
        Validate that the model is being loaded from a trusted source.

        Args:
            path: Path or URL to the model file

        Raises:
            ValueError: If the source is not trusted
        """
        if not path or not path.strip():
            raise ValueError("Model path must not be empty.")

        # Get allowed model storage locations from environment
        allowed_s3_buckets = os.getenv("ALLOWED_S3_BUCKETS", "").split(",")

        # For URLs, validate against allowed S3 buckets
        if path.startswith(("http://", "https://")):
            parsed = urlparse(path)
            hostname = (parsed.hostname or "").lower()

            # Check if it's an S3 URL
            is_s3_host = hostname.endswith(".amazonaws.com") or hostname == "amazonaws.com"
            if is_s3_host and allowed_s3_buckets and allowed_s3_buckets[0]:
                # Extract bucket name from URL
                bucket_name = self._extract_s3_bucket(hostname, parsed.path)
                if not bucket_name or bucket_name not in [b for b in allowed_s3_buckets if b]:
                    logger.warning("Attempted to load model from untrusted S3 bucket: %s", hostname)
                    raise ValueError("Model source not trusted. Only models from configured S3 buckets are allowed.")
            # Allow other HTTPS sources (pre-signed URLs validated by authentication)
            return

        # For local files, validate against allowed directories
        allowed_dirs = os.getenv("ALLOWED_MODEL_DIRS", "").split(",")
        if allowed_dirs and allowed_dirs[0]:
            safe_filename = os.path.basename(os.path.normpath(path))
            if not safe_filename or safe_filename in (".", ".."):
                raise ValueError(f"Invalid model filename: {path}")

            is_allowed = False
            for allowed_dir in allowed_dirs:
                if not allowed_dir:
                    continue
                allowed_base = str(Path(allowed_dir).resolve())
                candidate_path = os.path.normpath(os.path.join(allowed_base, safe_filename))
                if candidate_path.startswith(allowed_base + os.sep):
                    is_allowed = True
                    break

            if not is_allowed:
                raise ValueError("Model source not trusted. Only models from configured directories are allowed.")

    def _extract_s3_bucket(self, host: str, url_path: str) -> str | None:
        """Extract S3 bucket name from URL."""
        host = host or ""
        url_path = (url_path or "").lstrip("/")

        # Virtual-hosted style: <bucket>.s3[.<region>].amazonaws.com
        host_parts = host.split(".")
        if len(host_parts) >= 3 and host_parts[-2] == "amazonaws" and host_parts[-1] == "com":
            bucket_candidate = host_parts[0]
            if bucket_candidate and bucket_candidate not in ("s3", "s3-accelerate"):
                return bucket_candidate

        # Path-style: s3[.<region>].amazonaws.com/<bucket>/...
        if len(host_parts) >= 3 and host_parts[0].startswith("s3"):
            if url_path:
                path_parts = url_path.split("/")
                bucket_candidate = path_parts[0]
                if bucket_candidate:
                    return bucket_candidate

        return None

    def _validate_external_url(self, url: str) -> None:
        """
        Validate external URL to prevent SSRF attacks.

        Args:
            url: URL to validate

        Raises:
            ValueError: If URL is potentially unsafe
        """
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP/HTTPS URLs are allowed")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("URL must include a valid hostname")

        try:
            # Resolve hostname to IP addresses
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve hostname '{hostname}': {exc}") from exc

        # Check all resolved IPs
        ip_addresses = []
        for family, _, _, _, sockaddr in addr_info:
            if family not in (socket.AF_INET, socket.AF_INET6):
                continue

            ip_str = sockaddr[0]
            ip_addresses.append(ip_str)

            try:
                ip_obj = ipaddress.ip_address(ip_str)

                # Block private addresses
                if ip_obj.is_private:
                    raise ValueError(
                        f"Access to private addresses is not allowed. "
                        f"Hostname '{hostname}' resolves to private IP: {ip_str}"
                    )

                # Block loopback addresses
                if ip_obj.is_loopback:
                    raise ValueError(
                        f"Access to loopback addresses is not allowed. "
                        f"Hostname '{hostname}' resolves to loopback IP: {ip_str}"
                    )

                # Block link-local addresses
                if ip_obj.is_link_local:
                    raise ValueError(
                        f"Access to link-local addresses is not allowed. "
                        f"Hostname '{hostname}' resolves to link-local IP: {ip_str}"
                    )

                # Block multicast addresses
                if ip_obj.is_multicast:
                    raise ValueError(
                        f"Access to multicast addresses is not allowed. "
                        f"Hostname '{hostname}' resolves to multicast IP: {ip_str}"
                    )

                # Block reserved addresses
                if ip_obj.is_reserved:
                    raise ValueError(
                        f"Access to reserved addresses is not allowed. "
                        f"Hostname '{hostname}' resolves to reserved IP: {ip_str}"
                    )

                # Block unspecified addresses (0.0.0.0, ::)
                if ip_obj.is_unspecified:
                    raise ValueError(
                        f"Access to unspecified addresses is not allowed. "
                        f"Hostname '{hostname}' resolves to unspecified IP: {ip_str}"
                    )

            except ValueError as exc:
                # Re-raise our custom validation errors
                if "not allowed" in str(exc):
                    raise
                raise ValueError(f"Invalid IP address format: {ip_str}") from exc

        logger.info("URL validation passed for: %s (resolved to: %s)", hostname, ip_addresses)

    def _validate_external_url_with_allowlist(self, url: str) -> None:
        """
        Validate dataset URL with allow-list support.

        Args:
            url: URL to validate

        Raises:
            ValueError: If URL is potentially unsafe
        """
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP/HTTPS URLs are allowed for datasets")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Dataset URL must include a valid hostname")

        # Optional hostname allow-list
        allowed_hosts_env = os.getenv("EXPLAINABILITY_ALLOWED_HOSTS", "").strip()
        if allowed_hosts_env:
            allowed_hosts = {h.strip().lower() for h in allowed_hosts_env.split(",") if h.strip()}
        else:
            allowed_hosts = set()

        host_lower = hostname.lower()
        if allowed_hosts:

            def _is_allowed(h: str) -> bool:
                for allowed in allowed_hosts:
                    if h == allowed or h.endswith("." + allowed):
                        return True
                return False

            if not _is_allowed(host_lower):
                raise ValueError(f"Host '{hostname}' is not in the list of allowed dataset hosts")

        # Validate resolved IPs
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve dataset host '{hostname}': {exc}") from exc

        for family, _, _, _, sockaddr in addr_info:
            if family not in (socket.AF_INET, socket.AF_INET6):
                continue

            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            ):
                raise ValueError(
                    f"Dataset host '{hostname}' resolves to disallowed IP: {ip_str} "
                    "(private, loopback, link-local, multicast, reserved, or unspecified)"
                )

    def _load_dataset(self, dataset_url: str) -> pd.DataFrame:
        """
        Load dataset from URL with security validation.

        Args:
            dataset_url: URL to the dataset file

        Returns:
            pandas DataFrame

        Raises:
            ValueError: If dataset loading fails
        """
        if not dataset_url:
            raise ValueError("Empty dataset URL provided")

        # Security: Validate URL to prevent SSRF using centralized helper
        try:
            validated_url = validate_and_resolve_url(
                url=dataset_url,
                allowed_schemes={"http", "https"},
                allowed_hosts=get_default_allowed_hosts(),
                allowed_host_suffixes=get_default_allowed_host_suffixes(),
                source_name="dataset",
            )

            logger.info("Downloading dataset from: %s", dataset_url)
            # Use make_safe_request to prevent SSRF attacks via redirects
            response = make_safe_request(validated_url, timeout=30)

            # Determine file type from URL
            url_no_query = dataset_url.split("?", 1)[0]
            ext = os.path.splitext(url_no_query)[1].lower()

            if ext == ".csv":
                return pd.read_csv(io.StringIO(response.text))
            elif ext == ".parquet":
                return pd.read_parquet(io.BytesIO(response.content), engine="pyarrow")
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .parquet")

        except requests.exceptions.Timeout as exc:
            raise ValueError("Dataset download timed out") from exc
        except requests.exceptions.RequestException as exc:
            raise ValueError(f"Failed to download dataset: {exc}") from exc

    def _load_model(self, model_url: str) -> ModelWrapper:
        """
        Load model from URL with comprehensive security validation.

        Args:
            model_url: URL to the model file

        Returns:
            ModelWrapper containing the loaded model

        Raises:
            ValueError: If model loading fails or source is untrusted
        """
        # Security: Set file size limit (500 MB)
        MAX_MODEL_SIZE = 500 * 1024 * 1024  # 500 MB

        try:
            # Security: Validate URL using centralized helper with HTTPS-only and S3 allowlist
            logger.info("Validating and downloading model from: %s", model_url)
            validated_url = validate_and_resolve_url(
                url=model_url,
                allowed_schemes={"https"},  # HTTPS only for model downloads
                allowed_hosts=get_default_allowed_hosts(),
                allowed_host_suffixes=[".s3.amazonaws.com", ".amazonaws.com"],
                source_name="model",
            )

            # Use make_safe_request to prevent SSRF attacks via redirects
            response = make_safe_request(validated_url, timeout=30)

            # Check content length
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_MODEL_SIZE:
                raise ValueError(f"Model file too large: {content_length} bytes (max: {MAX_MODEL_SIZE})")

            model_bytes = response.content

            # Validate downloaded size
            if len(model_bytes) > MAX_MODEL_SIZE:
                raise ValueError(f"Model file too large: {len(model_bytes)} bytes (max: {MAX_MODEL_SIZE})")

            logger.info("Downloaded %s bytes", len(model_bytes))

            # Determine file extension from original URL
            parsed_url = urlparse(model_url)
            file_path = parsed_url.path
            file_extension = file_path.lower()

            # Load based on file extension
            if file_extension.endswith(".joblib"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                    tmp_file.write(model_bytes)
                    tmp_path = tmp_file.name

                try:
                    # Security: Use restricted unpickler for joblib
                    original_unpickler = pickle.Unpickler
                    try:
                        pickle.Unpickler = RestrictedUnpickler
                        model = joblib.load(tmp_path)
                    finally:
                        pickle.Unpickler = original_unpickler
                    return ModelWrapper(model, "sklearn")
                finally:
                    os.unlink(tmp_path)

            elif file_extension.endswith((".pkl", ".pickle")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                    tmp_file.write(model_bytes)
                    tmp_path = tmp_file.name

                try:
                    # Security: Use restricted unpickler
                    with open(tmp_path, "rb") as f:
                        model = RestrictedUnpickler(f).load()
                    return ModelWrapper(model, "sklearn")
                finally:
                    os.unlink(tmp_path)

            elif file_extension.endswith(".onnx"):
                if not ONNX_AVAILABLE:
                    raise ValueError(
                        "ONNX model format detected but onnxruntime is not installed. "
                        "Please install onnxruntime: pip install onnxruntime"
                    )
                # Load ONNX model directly from bytes
                onnx_session = ort.InferenceSession(model_bytes)
                logger.info("ONNX model loaded successfully")
                return ModelWrapper(onnx_session, "onnx")

            else:
                raise ValueError(
                    f"Unsupported model format in URL: {file_path}. " "Supported formats: .joblib, .pkl, .pickle, .onnx"
                )

        except requests.exceptions.Timeout as exc:
            raise ValueError("Model download timed out after 30 seconds") from exc
        except requests.exceptions.RequestException as exc:
            raise ValueError(f"Failed to download model: {str(exc)}") from exc

    def detect_model_type(self, model_wrapper: ModelWrapper, user_override: Optional[str] = None) -> Tuple[str, str]:
        """
        Detect if model is regression or classification.

        Args:
            model_wrapper: Loaded model wrapper object
            user_override: Optional user-specified model type ("regression" or "classification")

        Returns:
            Tuple of (model_type, confidence)
            model_type: "Regression" | "Classification" | "Unknown"
            confidence: "high" | "medium" | "low" | "user_specified"
        """
        # If user explicitly specified model type, use that
        if user_override:
            if user_override.lower() == "regression":
                return ("Regression", "user_specified")
            elif user_override.lower() == "classification":
                return ("Classification", "user_specified")

        try:
            model = model_wrapper.model

            # For ONNX models, we need different detection logic
            if model_wrapper.model_type == "onnx":
                # Check ONNX model outputs - classifiers typically have 2 outputs (labels + probabilities)
                outputs = model.get_outputs()
                if len(outputs) >= 2:
                    return ("Classification", "medium")
                # Check output name hints
                for output in outputs:
                    output_name = output.name.lower()
                    if any(hint in output_name for hint in ["label", "class", "prob"]):
                        return ("Classification", "medium")
                # Default to Unknown for ONNX without clear indicators
                return ("Unknown", "low")

            # Check for _estimator_type attribute (sklearn convention)
            if hasattr(model, "_estimator_type"):
                estimator_type = model._estimator_type
                if estimator_type == "regressor":
                    return ("Regression", "high")
                elif estimator_type == "classifier":
                    return ("Classification", "high")

            # Check class name patterns
            model_class = type(model).__name__.lower()

            regression_patterns = [
                "regressor",
                "regression",
                "svr",
                "ridge",
                "lasso",
                "elasticnet",
                "linearregression",
                "decisiontreeregressor",
                "randomforestregressor",
                "gradientboostingregressor",
                "xgbregressor",
                "lgbmregressor",
                "catboostregressor",
            ]

            classification_patterns = [
                "classifier",
                "classification",
                "svc",
                "logistic",
                "logisticregression",
                "decisiontreeclassifier",
                "randomforestclassifier",
                "gradientboostingclassifier",
                "xgbclassifier",
                "lgbmclassifier",
                "catboostclassifier",
            ]

            # Check for exact or partial matches
            for pattern in regression_patterns:
                if pattern in model_class:
                    return ("Regression", "high" if pattern == model_class else "medium")

            for pattern in classification_patterns:
                if pattern in model_class:
                    return ("Classification", "high" if pattern == model_class else "medium")

            # Check for predict_proba method (classification indicator)
            if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
                return ("Classification", "medium")

            # Check for predict method only (could be either)
            if hasattr(model, "predict") and callable(getattr(model, "predict")):
                return ("Unknown", "low")

            return ("Unknown", "low")

        except Exception as exc:
            logger.error("Error detecting model type: %s", exc)
            return ("Unknown", "low")

    def _calculate_regression_metrics(self, y_true, y_pred, n_features: int = 1) -> Dict[str, float]:
        """
        Calculate comprehensive regression performance metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            n_features: Number of features used in the model

        Returns:
            Dictionary with all regression metrics (r2_score, rmse, mse, mae, mape, smape, adjusted_r2, explained_variance)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:  # pylint: disable=broad-except
            # Fallback calculation if sklearn version doesn't support it or division by zero
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1)))

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean(np.where(denominator == 0, 0, 2.0 * np.abs(y_pred - y_true) / denominator)) * 100

        # Adjusted R²
        n_samples = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1) if n_samples > n_features + 1 else r2

        # Explained Variance
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0.0

        return {
            "r2_score": float(r2),
            "rmse": float(rmse),
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape),
            "smape": float(smape),
            "adjusted_r2": float(adj_r2),
            "explained_variance": float(explained_variance),
        }

    def _calculate_overfitting_score(self, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> float:
        """
        Calculate overfitting score based on train/test metric differences.

        Args:
            train_metrics: Training set metrics
            test_metrics: Test set metrics

        Returns:
            Overfitting score (0-1, lower is better)
        """
        # For regression: use R² difference
        if "r2_score" in train_metrics and "r2_score" in test_metrics:
            overfitting_score = max(0.0, train_metrics["r2_score"] - test_metrics["r2_score"])
        # For classification: use accuracy difference
        elif "accuracy" in train_metrics and "accuracy" in test_metrics:
            overfitting_score = max(0.0, train_metrics["accuracy"] - test_metrics["accuracy"])
        else:
            overfitting_score = 0.0

        return float(overfitting_score)

    def _calculate_classification_metrics(
        self, y_true, y_pred, y_proba: Optional[np.ndarray] = None, averaging: str = "macro"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification performance metrics.

        Args:
            y_true: True target labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for ROC-AUC)
            averaging: Averaging strategy for multi-class metrics ('macro' or 'weighted')

        Returns:
            Dictionary with all classification metrics (accuracy, precision, recall, f1_score, roc_auc)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Determine if binary or multiclass
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)
        is_binary = n_classes == 2

        # Accuracy (does not need averaging)
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 with specified averaging
        # Use zero_division=0 to handle cases with no positive predictions
        precision = precision_score(y_true, y_pred, average=averaging, zero_division=0)
        recall = recall_score(y_true, y_pred, average=averaging, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=averaging, zero_division=0)

        # ROC-AUC
        roc_auc = 0.0
        try:
            if y_proba is not None:
                if is_binary:
                    # Binary classification - use probability of positive class
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                    elif y_proba.ndim == 1:
                        roc_auc = roc_auc_score(y_true, y_proba)
                    else:
                        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multiclass - use One-vs-Rest (OVR) strategy
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=averaging)
            else:
                # If no probabilities, log warning and set to 0
                logger.warning("No prediction probabilities provided for ROC-AUC calculation")
        except (ValueError, RuntimeError) as exc:  # pylint: disable=broad-except
            logger.warning("Could not calculate ROC-AUC: %s", exc)
            roc_auc = 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
        }

    def evaluate_metric(
        self,
        metric_name: str,
        value: float,
        threshold_good: float,
        threshold_acceptable: float,
        higher_is_better: bool = False,
    ) -> MetricDetail:
        """
        Evaluate a metric against thresholds.

        Args:
            metric_name: Name of the metric
            value: Actual metric value
            threshold_good: Threshold for good performance
            threshold_acceptable: Threshold for acceptable performance
            higher_is_better: Whether higher values are better (True for R², False for errors)

        Returns:
            MetricDetail with status and color
        """
        if higher_is_better:
            # For R²: higher is better
            if value >= threshold_good:
                status = "good"
                color = "green"
            elif value >= threshold_acceptable:
                status = "acceptable"
                color = "yellow"
            else:
                status = "poor"
                color = "red"
        else:
            # For MAE, MSE, RMSE: lower is better
            if value <= threshold_good:
                status = "good"
                color = "green"
            elif value <= threshold_acceptable:
                status = "acceptable"
                color = "yellow"
            else:
                status = "poor"
                color = "red"

        return MetricDetail(
            value=value,
            threshold_good=threshold_good,
            threshold_acceptable=threshold_acceptable,
            status=status,
            color=color,
        )

    def _build_performance_summary(
        self, request: PerformanceSummaryRequest, metrics: RegressionPerformanceMetrics
    ) -> str:
        """
        Build text summary for LLM prompt (Regression).

        Args:
            request: Original request
            metrics: Regression performance metrics with train/test split

        Returns:
            Formatted text summary
        """
        summary = "=== Model Type: Regression ===\n\n"

        summary += "=== Training Set Performance ===\n\n"

        # Training MAE
        mae = metrics.train.mae
        summary += f"Mean Absolute Error (MAE): {mae.value:.6f}\n"
        summary += f"  - Status: {mae.status.upper()} ({mae.color})\n"
        summary += f"  - Thresholds: Good ≤ {mae.threshold_good}, Acceptable ≤ {mae.threshold_acceptable}\n\n"

        # Training MSE
        mse = metrics.train.mse
        summary += f"Mean Squared Error (MSE): {mse.value:.6f}\n"
        summary += f"  - Status: {mse.status.upper()} ({mse.color})\n"
        summary += f"  - Thresholds: Good ≤ {mse.threshold_good}, Acceptable ≤ {mse.threshold_acceptable}\n\n"

        # Training RMSE
        rmse = metrics.train.rmse
        summary += f"Root Mean Squared Error (RMSE): {rmse.value:.6f}\n"
        summary += f"  - Status: {rmse.status.upper()} ({rmse.color})\n"
        summary += f"  - Thresholds: Good ≤ {rmse.threshold_good}, Acceptable ≤ {rmse.threshold_acceptable}\n\n"

        # Training R²
        r2 = metrics.train.r_squared
        summary += f"R² (Coefficient of Determination): {r2.value:.6f}\n"
        summary += f"  - Status: {r2.status.upper()} ({r2.color})\n"
        summary += f"  - Thresholds: Good ≥ {r2.threshold_good}, Acceptable ≥ {r2.threshold_acceptable}\n\n"

        # Training MAPE
        mape = metrics.train.mape
        summary += f"Mean Absolute Percentage Error (MAPE): {mape.value:.6f}\n"
        summary += f"  - Status: {mape.status.upper()} ({mape.color})\n"
        summary += f"  - Thresholds: Good ≤ {mape.threshold_good}, Acceptable ≤ {mape.threshold_acceptable}\n\n"

        # Training SMAPE
        smape = metrics.train.smape
        summary += f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape.value:.6f}%\n"
        summary += f"  - Status: {smape.status.upper()} ({smape.color})\n"
        summary += f"  - Thresholds: Good ≤ {smape.threshold_good}%, Acceptable ≤ {smape.threshold_acceptable}%\n\n"

        # Training Adjusted R²
        adj_r2 = metrics.train.adjusted_r2
        summary += f"Adjusted R²: {adj_r2.value:.6f}\n"
        summary += f"  - Status: {adj_r2.status.upper()} ({adj_r2.color})\n"
        summary += f"  - Thresholds: Good ≥ {adj_r2.threshold_good}, Acceptable ≥ {adj_r2.threshold_acceptable}\n\n"

        # Training Explained Variance
        exp_var = metrics.train.explained_variance
        summary += f"Explained Variance: {exp_var.value:.6f}\n"
        summary += f"  - Status: {exp_var.status.upper()} ({exp_var.color})\n"
        summary += f"  - Thresholds: Good ≥ {exp_var.threshold_good}, Acceptable ≥ {exp_var.threshold_acceptable}\n\n"

        summary += "=== Test Set Performance ===\n\n"

        # Test MAE
        mae_test = metrics.test.mae
        summary += f"Mean Absolute Error (MAE): {mae_test.value:.6f}\n"
        summary += f"  - Status: {mae_test.status.upper()} ({mae_test.color})\n"
        summary += f"  - Thresholds: Good ≤ {mae_test.threshold_good}, Acceptable ≤ {mae_test.threshold_acceptable}\n\n"

        # Test MSE
        mse_test = metrics.test.mse
        summary += f"Mean Squared Error (MSE): {mse_test.value:.6f}\n"
        summary += f"  - Status: {mse_test.status.upper()} ({mse_test.color})\n"
        summary += f"  - Thresholds: Good ≤ {mse_test.threshold_good}, Acceptable ≤ {mse_test.threshold_acceptable}\n\n"

        # Test RMSE
        rmse_test = metrics.test.rmse
        summary += f"Root Mean Squared Error (RMSE): {rmse_test.value:.6f}\n"
        summary += f"  - Status: {rmse_test.status.upper()} ({rmse_test.color})\n"
        summary += (
            f"  - Thresholds: Good ≤ {rmse_test.threshold_good}, Acceptable ≤ {rmse_test.threshold_acceptable}\n\n"
        )

        # Test R²
        r2_test = metrics.test.r_squared
        summary += f"R² (Coefficient of Determination): {r2_test.value:.6f}\n"
        summary += f"  - Status: {r2_test.status.upper()} ({r2_test.color})\n"
        summary += f"  - Thresholds: Good ≥ {r2_test.threshold_good}, Acceptable ≥ {r2_test.threshold_acceptable}\n\n"

        # Test MAPE
        mape_test = metrics.test.mape
        summary += f"Mean Absolute Percentage Error (MAPE): {mape_test.value:.6f}\n"
        summary += f"  - Status: {mape_test.status.upper()} ({mape_test.color})\n"
        summary += (
            f"  - Thresholds: Good ≤ {mape_test.threshold_good}, Acceptable ≤ {mape_test.threshold_acceptable}\n\n"
        )

        # Test SMAPE
        smape_test = metrics.test.smape
        summary += f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_test.value:.6f}%\n"
        summary += f"  - Status: {smape_test.status.upper()} ({smape_test.color})\n"
        summary += (
            f"  - Thresholds: Good ≤ {smape_test.threshold_good}%, Acceptable ≤ {smape_test.threshold_acceptable}%\n\n"
        )

        # Test Adjusted R²
        adj_r2_test = metrics.test.adjusted_r2
        summary += f"Adjusted R²: {adj_r2_test.value:.6f}\n"
        summary += f"  - Status: {adj_r2_test.status.upper()} ({adj_r2_test.color})\n"
        summary += (
            f"  - Thresholds: Good ≥ {adj_r2_test.threshold_good}, Acceptable ≥ {adj_r2_test.threshold_acceptable}\n\n"
        )

        # Test Explained Variance
        exp_var_test = metrics.test.explained_variance
        summary += f"Explained Variance: {exp_var_test.value:.6f}\n"
        summary += f"  - Status: {exp_var_test.status.upper()} ({exp_var_test.color})\n"
        summary += f"  - Thresholds: Good ≥ {exp_var_test.threshold_good}, Acceptable ≥ {exp_var_test.threshold_acceptable}\n\n"

        # Overfitting Analysis
        summary += "=== Overfitting Analysis ===\n\n"
        summary += f"Overfitting Score: {metrics.overfitting_score:.4f} (0 = no overfitting, 1 = severe overfitting)\n"

        if metrics.overfitting_score < 0.1:
            summary += "Assessment: Minimal overfitting detected\n\n"
        elif metrics.overfitting_score < 0.3:
            summary += "Assessment: Moderate overfitting detected\n\n"
        else:
            summary += "Assessment: Significant overfitting detected - model may not generalize well\n\n"

        # Evaluation Context
        summary += "=== Evaluation Context ===\n\n"
        summary += f"Model URL: {request.model_url}\n"
        summary += f"Train Dataset: {request.train_dataset_url}\n"
        summary += f"Test Dataset: {request.test_dataset_url}\n"
        if request.target_column:
            summary += f"Target Column: {request.target_column}\n"

        return summary

    def _build_classification_performance_summary(
        self, request: PerformanceSummaryRequest, metrics: ClassificationPerformanceMetrics
    ) -> str:
        """
        Build text summary for LLM prompt (Classification).

        Args:
            request: Original request
            metrics: Classification performance metrics with train/test split

        Returns:
            Formatted text summary
        """
        summary = "=== Model Type: Classification ===\n\n"
        summary += f"=== Averaging Strategy: {request.averaging_strategy.upper()} ===\n\n"

        summary += "=== Training Set Performance ===\n\n"

        # Training Accuracy
        acc = metrics.train.accuracy
        summary += f"Accuracy: {acc.value:.6f}\n"
        summary += f"  - Status: {acc.status.upper()} ({acc.color})\n"
        summary += f"  - Thresholds: Good ≥ {acc.threshold_good}, Acceptable ≥ {acc.threshold_acceptable}\n\n"

        # Training Precision
        prec = metrics.train.precision
        summary += f"Precision ({request.averaging_strategy}): {prec.value:.6f}\n"
        summary += f"  - Status: {prec.status.upper()} ({prec.color})\n"
        summary += f"  - Thresholds: Good ≥ {prec.threshold_good}, Acceptable ≥ {prec.threshold_acceptable}\n\n"

        # Training Recall
        rec = metrics.train.recall
        summary += f"Recall ({request.averaging_strategy}): {rec.value:.6f}\n"
        summary += f"  - Status: {rec.status.upper()} ({rec.color})\n"
        summary += f"  - Thresholds: Good ≥ {rec.threshold_good}, Acceptable ≥ {rec.threshold_acceptable}\n\n"

        # Training F1 Score
        f1 = metrics.train.f1_score
        summary += f"F1 Score ({request.averaging_strategy}): {f1.value:.6f}\n"
        summary += f"  - Status: {f1.status.upper()} ({f1.color})\n"
        summary += f"  - Thresholds: Good ≥ {f1.threshold_good}, Acceptable ≥ {f1.threshold_acceptable}\n\n"

        # Training ROC-AUC
        roc = metrics.train.roc_auc
        summary += f"ROC-AUC: {roc.value:.6f}\n"
        summary += f"  - Status: {roc.status.upper()} ({roc.color})\n"
        summary += f"  - Thresholds: Good ≥ {roc.threshold_good}, Acceptable ≥ {roc.threshold_acceptable}\n\n"

        summary += "=== Test Set Performance ===\n\n"

        # Test Accuracy
        acc_test = metrics.test.accuracy
        summary += f"Accuracy: {acc_test.value:.6f}\n"
        summary += f"  - Status: {acc_test.status.upper()} ({acc_test.color})\n"
        summary += f"  - Thresholds: Good ≥ {acc_test.threshold_good}, Acceptable ≥ {acc_test.threshold_acceptable}\n\n"

        # Test Precision
        prec_test = metrics.test.precision
        summary += f"Precision ({request.averaging_strategy}): {prec_test.value:.6f}\n"
        summary += f"  - Status: {prec_test.status.upper()} ({prec_test.color})\n"
        summary += (
            f"  - Thresholds: Good ≥ {prec_test.threshold_good}, Acceptable ≥ {prec_test.threshold_acceptable}\n\n"
        )

        # Test Recall
        rec_test = metrics.test.recall
        summary += f"Recall ({request.averaging_strategy}): {rec_test.value:.6f}\n"
        summary += f"  - Status: {rec_test.status.upper()} ({rec_test.color})\n"
        summary += f"  - Thresholds: Good ≥ {rec_test.threshold_good}, Acceptable ≥ {rec_test.threshold_acceptable}\n\n"

        # Test F1 Score
        f1_test = metrics.test.f1_score
        summary += f"F1 Score ({request.averaging_strategy}): {f1_test.value:.6f}\n"
        summary += f"  - Status: {f1_test.status.upper()} ({f1_test.color})\n"
        summary += f"  - Thresholds: Good ≥ {f1_test.threshold_good}, Acceptable ≥ {f1_test.threshold_acceptable}\n\n"

        # Test ROC-AUC
        roc_test = metrics.test.roc_auc
        summary += f"ROC-AUC: {roc_test.value:.6f}\n"
        summary += f"  - Status: {roc_test.status.upper()} ({roc_test.color})\n"
        summary += f"  - Thresholds: Good ≥ {roc_test.threshold_good}, Acceptable ≥ {roc_test.threshold_acceptable}\n\n"

        # Overfitting Analysis
        summary += "=== Overfitting Analysis ===\n\n"
        summary += f"Overfitting Score: {metrics.overfitting_score:.4f} (0 = no overfitting, 1 = severe overfitting)\n"

        if metrics.overfitting_score < 0.1:
            summary += "Assessment: Minimal overfitting detected\n\n"
        elif metrics.overfitting_score < 0.3:
            summary += "Assessment: Moderate overfitting detected\n\n"
        else:
            summary += "Assessment: Significant overfitting detected - model may not generalize well\n\n"

        # Evaluation Context
        summary += "=== Evaluation Context ===\n\n"
        summary += f"Model URL: {request.model_url}\n"
        summary += f"Train Dataset: {request.train_dataset_url}\n"
        summary += f"Test Dataset: {request.test_dataset_url}\n"
        if request.target_column:
            summary += f"Target Column: {request.target_column}\n"

        return summary

    def generate_interpretation(
        self,
        request: PerformanceSummaryRequest,
        metrics: Union[RegressionPerformanceMetrics, ClassificationPerformanceMetrics],
        model_type: str,
    ) -> PerformanceInterpretation:
        """
        Generate LLM-based interpretation of performance results.

        Args:
            request: Original request
            metrics: Performance metrics with train/test split
            model_type: "Regression" or "Classification"

        Returns:
            PerformanceInterpretation with what_this_means, why_it_matters, and risk_signal
        """
        # Build summary for LLM based on model type
        if model_type == "Classification":
            summary_text = self._build_classification_performance_summary(request, metrics)
            model_type_str = "classification"
            key_metrics_str = "Accuracy, Precision, Recall, F1 Score, and ROC-AUC"
        else:
            summary_text = self._build_performance_summary(request, metrics)
            model_type_str = "regression"
            key_metrics_str = "R², MAE, MSE, RMSE, and error metrics"

        prompt = f"""You are an expert in machine learning performance evaluation and model deployment fitness assessment. Analyze the following {model_type_str} model performance metrics:

{summary_text}

Provide a comprehensive yet concise analysis structured in THREE sections:

1. **What This Means** (2-3 sentences)
   Provide a plain-language explanation of the model's predictive quality. Explain what the key metrics ({key_metrics_str}) indicate about the model's performance. Compare train vs test results.

2. **Why It Matters** (2-3 sentences)
   Explain the real-world implications and business impact of these metrics. What does this mean for model deployment? How reliable are the predictions?

3. **Risk Signal** (1-2 sentences)
   Provide an overall risk assessment. Classify as "Low Risk", "Medium Risk", or "High Risk" and explain why. Focus on overfitting, metric degradation, or threshold breaches.

Guidelines:
- Use clear, informational language suitable for ML practitioners and business stakeholders
- Be non-prescriptive - describe what the metrics indicate, not what actions to take
- Reference specific metric values and their status colors in your analysis
- Pay special attention to the overfitting score and train/test differences
- Keep language concise and impactful

Format your response EXACTLY as:

**What This Means:**
[Your 2-3 sentence explanation]

**Why It Matters:**
[Your 2-3 sentence explanation]

**Risk Signal:**
[Your 1-2 sentence risk assessment]"""

        # Invoke Claude
        analysis_text = self._invoke_claude(prompt)

        # Parse the response
        return self._parse_interpretation(analysis_text, metrics)

    def _parse_interpretation(
        self, analysis_text: str, metrics: Union[RegressionPerformanceMetrics, ClassificationPerformanceMetrics]
    ) -> PerformanceInterpretation:
        """
        Parse LLM response into structured interpretation (Fairness-aligned format).

        Args:
            analysis_text: Raw text from LLM
            metrics: Performance metrics for fallback logic

        Returns:
            PerformanceInterpretation object with what_this_means, why_it_matters, risk_signal
        """
        result = {"what_this_means": "", "why_it_matters": "", "risk_signal": ""}

        try:
            # Split into sections by ** markers
            sections = analysis_text.split("**")

            current_section = None
            for section in sections:
                section = section.strip()
                if not section:
                    continue

                section_lower = section.lower()

                # Identify section headers
                if "what this means" in section_lower or "what_this_means" in section_lower:
                    current_section = "what_this_means"
                    if ":" in section:
                        result["what_this_means"] = section.split(":", 1)[1].strip()
                elif "why it matters" in section_lower or "why_it_matters" in section_lower:
                    current_section = "why_it_matters"
                    if ":" in section:
                        result["why_it_matters"] = section.split(":", 1)[1].strip()
                elif "risk signal" in section_lower or "risk_signal" in section_lower:
                    current_section = "risk_signal"
                    if ":" in section:
                        result["risk_signal"] = section.split(":", 1)[1].strip()
                else:
                    # This is content for the current section
                    if current_section and not result[current_section]:
                        result[current_section] = section.strip()

            # Fallback values if parsing failed
            if not result["what_this_means"]:
                result["what_this_means"] = (
                    "Performance analysis completed. The model metrics have been evaluated "
                    "against user-defined thresholds. Review individual metric details for specific insights."
                )

            if not result["why_it_matters"]:
                result["why_it_matters"] = (
                    "Understanding model performance helps ensure reliable predictions in production. "
                    "Consistent train/test performance indicates good generalization capability."
                )

            if not result["risk_signal"]:
                # Generate risk signal based on overfitting score
                if metrics.overfitting_score < 0.1:
                    result["risk_signal"] = "Low Risk - Model metrics are healthy with minimal overfitting detected."
                elif metrics.overfitting_score < 0.3:
                    result["risk_signal"] = (
                        "Medium Risk - Moderate overfitting detected. Monitor performance on new data."
                    )
                else:
                    result["risk_signal"] = (
                        "High Risk - Significant overfitting detected. Model may not generalize well to new data."
                    )

            return PerformanceInterpretation(
                what_this_means=result["what_this_means"],
                why_it_matters=result["why_it_matters"],
                risk_signal=result["risk_signal"],
            )

        except Exception as exc:
            logger.error("Error parsing interpretation: %s", exc)
            # Return fallback interpretation
            return PerformanceInterpretation(
                what_this_means="Performance analysis completed. Review individual metrics for detailed assessment.",
                why_it_matters="Model performance metrics help assess deployment readiness and prediction reliability.",
                risk_signal="Review metrics manually - automated risk assessment encountered an issue.",
            )

    def calculate_performance_summary(self, request: PerformanceSummaryRequest) -> PerformanceSummaryResponse:
        """
        Calculate complete performance summary with model detection and LLM interpretation.
        Supports both Regression and Classification models.

        Args:
            request: Performance summary request

        Returns:
            PerformanceSummaryResponse with enriched data

        Raises:
            ValueError: If data processing fails
        """
        # Load and detect model type
        logger.info("Loading model from: %s", request.model_url)
        model_wrapper = self._load_model(str(request.model_url))

        logger.info("Detecting model type...")
        model_type, confidence = self.detect_model_type(model_wrapper, request.model_type)

        if model_type == "Unknown":
            logger.error(
                "Could not confidently detect model type (confidence: %s). "
                "Please specify 'model_type' explicitly as 'Regression' or 'Classification'.",
                confidence,
            )
            raise ValueError(
                "Unable to detect model type automatically. "
                "Specify 'model_type' as 'Regression' or 'Classification' in the request."
            )

        logger.info("Model type: %s (confidence: %s)", model_type, confidence)

        # Load datasets
        logger.info("Loading training dataset from: %s", request.train_dataset_url)
        train_df = self._load_dataset(str(request.train_dataset_url))

        logger.info("Loading test dataset from: %s", request.test_dataset_url)
        test_df = self._load_dataset(str(request.test_dataset_url))

        # Auto-detect target column if not provided or empty string
        target_column = request.target_column
        if not target_column or target_column.strip() == "":
            if train_df.empty or len(train_df.columns) == 0:
                raise ValueError("Training dataset is empty or has no columns")
            target_column = train_df.columns[-1]
            logger.info("Target column not provided. Auto-detected as last column: %s", target_column)

        # Validate target column exists
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training dataset")
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test dataset")

        # Extract features and targets
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        logger.info("Training samples: %s, Test samples: %s", len(X_train), len(X_test))

        # Make predictions
        logger.info("Making predictions on train and test sets...")
        y_train_pred = model_wrapper.predict(X_train)
        y_test_pred = model_wrapper.predict(X_test)

        # Route to appropriate analysis based on model type
        if model_type == "Classification":
            return self._calculate_classification_summary(
                request,
                model_wrapper,
                model_type,
                confidence,
                X_train,
                y_train,
                y_train_pred,
                X_test,
                y_test,
                y_test_pred,
            )
        else:
            return self._calculate_regression_summary(
                request, model_type, confidence, X_train, y_train, y_train_pred, X_test, y_test, y_test_pred
            )

    def _calculate_regression_summary(
        self,
        request: PerformanceSummaryRequest,
        model_type: str,
        confidence: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_train_pred: np.ndarray,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_test_pred: np.ndarray,
    ) -> PerformanceSummaryResponse:
        """Calculate regression-specific performance summary."""
        # Calculate metrics
        logger.info("Calculating regression performance metrics...")
        n_features = len(X_train.columns)
        train_metrics = self._calculate_regression_metrics(y_train, y_train_pred, n_features=n_features)
        test_metrics = self._calculate_regression_metrics(y_test, y_test_pred, n_features=n_features)

        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(train_metrics, test_metrics)

        logger.info(
            "Train R²: %.4f, Test R²: %.4f, Overfitting Score: %.4f",
            train_metrics["r2_score"],
            test_metrics["r2_score"],
            overfitting_score,
        )

        # Get thresholds
        thresholds = request.regression_thresholds

        # Evaluate metrics against thresholds
        logger.info("Evaluating metrics against thresholds...")
        train_metrics_output = RegressionMetricsOutput(
            mae=self.evaluate_metric(
                "MAE",
                train_metrics["mae"],
                thresholds.mae.good,
                thresholds.mae.acceptable,
                higher_is_better=False,
            ),
            mse=self.evaluate_metric(
                "MSE",
                train_metrics["mse"],
                thresholds.mse.good,
                thresholds.mse.acceptable,
                higher_is_better=False,
            ),
            rmse=self.evaluate_metric(
                "RMSE",
                train_metrics["rmse"],
                thresholds.rmse.good,
                thresholds.rmse.acceptable,
                higher_is_better=False,
            ),
            r_squared=self.evaluate_metric(
                "R²",
                train_metrics["r2_score"],
                thresholds.r_squared.good,
                thresholds.r_squared.acceptable,
                higher_is_better=True,
            ),
            mape=self.evaluate_metric(
                "MAPE",
                train_metrics["mape"],
                thresholds.mape.good,
                thresholds.mape.acceptable,
                higher_is_better=False,
            ),
            smape=self.evaluate_metric(
                "SMAPE",
                train_metrics["smape"],
                thresholds.smape.good,
                thresholds.smape.acceptable,
                higher_is_better=False,
            ),
            adjusted_r2=self.evaluate_metric(
                "Adjusted R²",
                train_metrics["adjusted_r2"],
                thresholds.adjusted_r2.good,
                thresholds.adjusted_r2.acceptable,
                higher_is_better=True,
            ),
            explained_variance=self.evaluate_metric(
                "Explained Variance",
                train_metrics["explained_variance"],
                thresholds.explained_variance.good,
                thresholds.explained_variance.acceptable,
                higher_is_better=True,
            ),
        )

        test_metrics_output = RegressionMetricsOutput(
            mae=self.evaluate_metric(
                "MAE",
                test_metrics["mae"],
                thresholds.mae.good,
                thresholds.mae.acceptable,
                higher_is_better=False,
            ),
            mse=self.evaluate_metric(
                "MSE",
                test_metrics["mse"],
                thresholds.mse.good,
                thresholds.mse.acceptable,
                higher_is_better=False,
            ),
            rmse=self.evaluate_metric(
                "RMSE",
                test_metrics["rmse"],
                thresholds.rmse.good,
                thresholds.rmse.acceptable,
                higher_is_better=False,
            ),
            r_squared=self.evaluate_metric(
                "R²",
                test_metrics["r2_score"],
                thresholds.r_squared.good,
                thresholds.r_squared.acceptable,
                higher_is_better=True,
            ),
            mape=self.evaluate_metric(
                "MAPE",
                test_metrics["mape"],
                thresholds.mape.good,
                thresholds.mape.acceptable,
                higher_is_better=False,
            ),
            smape=self.evaluate_metric(
                "SMAPE",
                test_metrics["smape"],
                thresholds.smape.good,
                thresholds.smape.acceptable,
                higher_is_better=False,
            ),
            adjusted_r2=self.evaluate_metric(
                "Adjusted R²",
                test_metrics["adjusted_r2"],
                thresholds.adjusted_r2.good,
                thresholds.adjusted_r2.acceptable,
                higher_is_better=True,
            ),
            explained_variance=self.evaluate_metric(
                "Explained Variance",
                test_metrics["explained_variance"],
                thresholds.explained_variance.good,
                thresholds.explained_variance.acceptable,
                higher_is_better=True,
            ),
        )

        # Create combined metrics object
        metrics = RegressionPerformanceMetrics(
            train=train_metrics_output, test=test_metrics_output, overfitting_score=overfitting_score
        )

        # Generate LLM interpretation
        logger.info("Generating LLM interpretation...")
        interpretation = self.generate_interpretation(request, metrics, model_type)

        # Auto-generate metadata
        from datetime import datetime, timezone

        test_dataset_name = os.path.basename(str(request.test_dataset_url).split("?")[0])
        model_name = os.path.basename(str(request.model_url).split("?")[0])

        metadata = PerformanceMetadata(
            asset_id=model_name.replace(".pkl", "").replace(".joblib", "").replace(".onnx", ""),
            asset_version="1.0.0",
            model_version="1.0.0",
            dataset_source=test_dataset_name,
            evaluation_date=datetime.now(timezone.utc).isoformat(),
        )

        return PerformanceSummaryResponse(
            model_type=model_type,
            model_detection_confidence=confidence,
            metrics=metrics,
            interpretation=interpretation,
            averaging_strategy=None,
            metadata=metadata,
        )

    def _calculate_classification_summary(
        self,
        request: PerformanceSummaryRequest,
        model_wrapper: ModelWrapper,
        model_type: str,
        confidence: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_train_pred: np.ndarray,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_test_pred: np.ndarray,
    ) -> PerformanceSummaryResponse:
        """Calculate classification-specific performance summary."""
        # Try to get prediction probabilities for ROC-AUC
        y_train_proba = None
        y_test_proba = None

        try:
            y_train_proba = model_wrapper.predict_proba(X_train)
            y_test_proba = model_wrapper.predict_proba(X_test)
            logger.info("Successfully obtained prediction probabilities for ROC-AUC")
        except AttributeError:
            logger.warning("Model does not support predict_proba. ROC-AUC will be set to 0.")
        except Exception as exc:
            logger.warning("Could not get prediction probabilities: %s", exc)

        # Calculate metrics
        logger.info("Calculating classification performance metrics...")
        averaging = request.averaging_strategy

        train_metrics = self._calculate_classification_metrics(y_train, y_train_pred, y_train_proba, averaging)
        test_metrics = self._calculate_classification_metrics(y_test, y_test_pred, y_test_proba, averaging)

        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(train_metrics, test_metrics)

        logger.info(
            "Train Accuracy: %.4f, Test Accuracy: %.4f, Overfitting Score: %.4f",
            train_metrics["accuracy"],
            test_metrics["accuracy"],
            overfitting_score,
        )

        # Get thresholds
        thresholds = request.classification_thresholds

        # Evaluate metrics against thresholds
        logger.info("Evaluating metrics against thresholds...")
        train_metrics_output = ClassificationMetricsOutput(
            accuracy=self.evaluate_metric(
                "Accuracy",
                train_metrics["accuracy"],
                thresholds.accuracy.good,
                thresholds.accuracy.acceptable,
                higher_is_better=True,
            ),
            precision=self.evaluate_metric(
                "Precision",
                train_metrics["precision"],
                thresholds.precision.good,
                thresholds.precision.acceptable,
                higher_is_better=True,
            ),
            recall=self.evaluate_metric(
                "Recall",
                train_metrics["recall"],
                thresholds.recall.good,
                thresholds.recall.acceptable,
                higher_is_better=True,
            ),
            f1_score=self.evaluate_metric(
                "F1 Score",
                train_metrics["f1_score"],
                thresholds.f1_score.good,
                thresholds.f1_score.acceptable,
                higher_is_better=True,
            ),
            roc_auc=self.evaluate_metric(
                "ROC-AUC",
                train_metrics["roc_auc"],
                thresholds.roc_auc.good,
                thresholds.roc_auc.acceptable,
                higher_is_better=True,
            ),
        )

        test_metrics_output = ClassificationMetricsOutput(
            accuracy=self.evaluate_metric(
                "Accuracy",
                test_metrics["accuracy"],
                thresholds.accuracy.good,
                thresholds.accuracy.acceptable,
                higher_is_better=True,
            ),
            precision=self.evaluate_metric(
                "Precision",
                test_metrics["precision"],
                thresholds.precision.good,
                thresholds.precision.acceptable,
                higher_is_better=True,
            ),
            recall=self.evaluate_metric(
                "Recall",
                test_metrics["recall"],
                thresholds.recall.good,
                thresholds.recall.acceptable,
                higher_is_better=True,
            ),
            f1_score=self.evaluate_metric(
                "F1 Score",
                test_metrics["f1_score"],
                thresholds.f1_score.good,
                thresholds.f1_score.acceptable,
                higher_is_better=True,
            ),
            roc_auc=self.evaluate_metric(
                "ROC-AUC",
                test_metrics["roc_auc"],
                thresholds.roc_auc.good,
                thresholds.roc_auc.acceptable,
                higher_is_better=True,
            ),
        )

        # Create combined metrics object
        metrics = ClassificationPerformanceMetrics(
            train=train_metrics_output, test=test_metrics_output, overfitting_score=overfitting_score
        )

        # Generate LLM interpretation
        logger.info("Generating LLM interpretation...")
        interpretation = self.generate_interpretation(request, metrics, model_type)

        # Auto-generate metadata
        from datetime import datetime, timezone

        test_dataset_name = os.path.basename(str(request.test_dataset_url).split("?")[0])
        model_name = os.path.basename(str(request.model_url).split("?")[0])

        metadata = PerformanceMetadata(
            asset_id=model_name.replace(".pkl", "").replace(".joblib", "").replace(".onnx", ""),
            asset_version="1.0.0",
            model_version="1.0.0",
            dataset_source=test_dataset_name,
            evaluation_date=datetime.now(timezone.utc).isoformat(),
        )

        return PerformanceSummaryResponse(
            model_type=model_type,
            model_detection_confidence=confidence,
            metrics=metrics,
            interpretation=interpretation,
            averaging_strategy=averaging,
            metadata=metadata,
        )
