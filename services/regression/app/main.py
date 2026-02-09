import json
import os
from typing import Dict, List, Optional

import requests
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.regression.app.core.ai_explanation_service import AIExplanationService
from services.regression.app.core.model_service import ModelService
from services.regression.app.database.connections import get_db
from services.regression.app.logging_config import logger
from services.regression.app.routers import regression
from services.regression.app.schemas.regression_schema import CorrelationPayload, RequestPayload
from services.regression.app.utils.analysisid import generate_analysis_id
from services.regression.app.utils.error_handler import handle_request
from shared.auth import get_current_user
from shared_migrations.models.analysis_result import AnalysisResult

app = FastAPI(
    title="Welcome to Regression Service",
    description="This service handles regression tasks, provides endpoints for model predictions, and includes performance and explainability analysis capabilities.",
    version="3.1.0",
    docs_url="/regression/docs",
    openapi_url="/regression/openapi.json",
    redoc_url="/regression/redocs",
)


@app.get("/regression/health", tags=["health"])
async def health_check():
    return {"status": "Regression healthy"}


# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:3000"] or your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- API Endpoints ---


@app.get("/regression/api/files", tags=["Mandatory"])
def get_s3_file_metadata(project_id: str, user: str = Depends(get_current_user)):
    """
    Lists files and models from the external S3 API and returns their metadata (name, URL, folder).
    Separates files and models based on the folder field.
    """
    file_api = os.getenv("FILES_API_BASE_URL")
    token = user.get("token")
    if not file_api:
        raise HTTPException(status_code=500, detail="FILES_API_BASE_URL environment variable is not set.")
    if not token:
        raise HTTPException(status_code=401, detail="User token is missing or invalid.")

    EXTERNAL_S3_API_URL = f"{file_api}/Regression/{project_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(EXTERNAL_S3_API_URL, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        json_data = response.json()
        all_items = json_data.get("files", [])

        # Separate files and models based on folder
        files = [item for item in all_items if item.get("folder") == "files"]
        models = [item for item in all_items if item.get("folder") == "models"]

        return {
            "success": True,
            "files": files,
            "models": models,
            "total_files": len(files),
            "total_models": len(models),
        }
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to external S3 API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to external S3 API: {str(e)}")
    except Exception as e:
        print(f"Error processing external S3 API response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing S3 API response: {str(e)}")


class LoadDataRequest(BaseModel):
    cur_dataset: Optional[str] = None
    ref_dataset: Optional[str] = None
    model: str
    target_column: str = "target"


@app.post("/regression/load", tags=["Mandatory"])
async def load_data(
    payload: LoadDataRequest,
    user: dict = Depends(get_current_user),
):
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication failed or user ID missing.")

    if not payload.model:
        raise HTTPException(status_code=400, detail="'model' path or URL is required.")

    if not payload.ref_dataset:
        raise HTTPException(status_code=400, detail="'ref_dataset' (training dataset) is required.")

    if not payload.cur_dataset:
        raise HTTPException(status_code=400, detail="'cur_dataset' (test dataset) is required.")

    if not payload.target_column:
        raise HTTPException(status_code=400, detail="'target_column' is required.")

    model_service = ModelService()

    try:
        return handle_request(
            model_service.load_model_and_datasets,
            model_path=payload.model,
            train_data_path=payload.ref_dataset,
            test_data_path=payload.cur_dataset,
            target_column=payload.target_column,
        )

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model or dataset file not found at the specified path.")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Dataset could not be loaded or parsed: {str(ve)}")

    except KeyError:
        raise HTTPException(
            status_code=400, detail=f"Target column '{payload.target_column}' does not exist in the dataset."
        )

    except TypeError as te:
        raise HTTPException(status_code=400, detail=f"Invalid input format provided: {str(te)}")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error while loading regression model or datasets: {str(e)}"
        )


@app.post("/regression/analysis/overview", tags=["Analysis"])
async def get_overview(payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)):

    try:
        user_id = user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="User authentication failed or user ID missing.")
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail="'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail="'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail="'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail="'target_column' is required.")

        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column

        model_service = ModelService()

        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )

        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:

            data_preview_response = handle_request(model_service.get_model_overview)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error retrieving regression model overview from model service: {str(e)}"
            )
        try:
            data_preview = json.loads(data_preview_response.body)
        except (ValueError, TypeError, json.JSONDecodeError):
            raise HTTPException(status_code=500, detail="Invalid JSON format received from regression model service.")
        try:
            analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating analysis ID: {str(e)}")
        try:

            existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Database query error while checking existing regression analysis: {str(e)}"
            )

        if not existing:

            try:
                new_entry = AnalysisResult(
                    analysis_id=analysis_id,
                    user_id=user_id,
                    analysis_type="regression",
                    analysis_tab="overview",
                    project_id="N/A",
                    json_result=json.dumps(data_preview),
                )
                db.add(new_entry)
                db.commit()
            except Exception as e:
                db.rollback()
                raise HTTPException(
                    status_code=500, detail=f"Error saving regression analysis overview to database: {str(e)}"
                )
            try:
                return handle_request(model_service.get_model_overview)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Error fetching final regression model overview: {str(e)}")

    except HTTPException:
        raise

    # ✅ Catch-all for any unexpected runtime errors
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error while processing regression analysis overview: {str(e)}"
        )


@app.post("/regression/analysis/regression-stats", tags=["Analysis"])
async def get_regression_statistics(
    payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:

        user_id = user.get("user_id") if user else None
        if not user_id:
            raise HTTPException(status_code=401, detail="User authentication failed or user ID missing.")

        # ✅ Validate payload fields
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail=" 'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail=" 'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail=" 'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")

        model_service = ModelService()

        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column

        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.get_regression_stats)
        except Exception as e:
            raise HTTPException(
                status_code=502, detail=f"Error retrieving regression statistics from model service: {str(e)}"
            )
        try:
            data_preview = json.loads(data_preview_response.body)
        except (ValueError, TypeError, json.JSONDecodeError):
            raise HTTPException(status_code=500, detail="Invalid JSON format received from regression model service.")
        try:
            analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating analysis ID for regression statistics: {str(e)}"
            )

        # Generate analysis_id
        try:
            existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Database query error while checking existing regression statistics: {str(e)}"
            )
        if not existing:
            try:

                new_entry = AnalysisResult(
                    analysis_id=analysis_id,
                    user_id=user_id,
                    analysis_type="regression",
                    analysis_tab="regression-stats",
                    project_id="N/A",
                    json_result=json.dumps(data_preview),
                )
                db.add(new_entry)
                db.commit()
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=f"Error saving regression statistics to database: {str(e)}")
            try:

                return handle_request(model_service.get_regression_stats)
            except Exception as e:
                raise HTTPException(
                    status_code=502, detail=f"Error fetching final regression statistics from model service: {str(e)}"
                )
    except HTTPException:
        raise

    # ✅ Catch-all for unexpected runtime errors
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error while processing regression statistics: {str(e)}"
        )


@app.post("/regression/analysis/feature-importance", tags=["Analysis"])
async def get_feature_importance(
    payload: RequestPayload, method: str = "shap", user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:

        model_service = ModelService()
        user_id = user.get("user_id")
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail=" 'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail=" 'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail=" 'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")

        if method.lower() not in ["shap", "permutation", "gain"]:
            raise HTTPException(
                status_code=400,
                detail=f" Unsupported feature importance method '{method}'. Supported: shap, permutation, gain.",
            )
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")

        try:

            data_preview_response = handle_request(
                model_service.compute_feature_importance_advanced, method, "importance", 1000, "bar"
            )
            data_preview = json.loads(data_preview_response.body)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to fetch feature importance: {str(e)}")

        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="feature-importance",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.compute_feature_importance_advanced, method, "importance", 1000, "bar")
    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f" Failed to generate feature importance due to an unexpected internal error: {str(e)}",
        )


@app.post("/regression/analysis/explain-instance/{instance_idx}", tags=["Analysis"])
async def explain_instance(
    payload: RequestPayload,
    instance_idx: int,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
    method: str = "shap",
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail="'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail="'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail="'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")
        if method.lower() not in ["shap", "lime", "integrated_gradients"]:
            raise HTTPException(
                status_code=400,
                detail=f" Unsupported explanation method '{method}'. Supported: shap, lime, integrated_gradients.",
            )
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column

        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.explain_instance, instance_idx)
            data_preview = json.loads(data_preview_response.body)
        except IndexError:
            raise HTTPException(status_code=400, detail=f" Instance index {instance_idx} is out of range.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to generate instance explanation: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="explain-instance",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.explain_instance, instance_idx)
    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f" Failed to explain instance due to an unexpected internal error: {str(e)}"
        )


@app.post("/regression/analysis/what-if", tags=["Analysis"])
async def perform_what_if(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        if not url_payload.ref_dataset:
            raise HTTPException(status_code=400, detail=" 'ref_dataset' (training dataset) is required.")
        if not url_payload.cur_dataset:
            raise HTTPException(status_code=400, detail=" 'cur_dataset' (test dataset) is required.")
        if not url_payload.model:
            raise HTTPException(status_code=400, detail=" 'model' path or URL is required.")
        if not url_payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")
        features = payload.get("features")
        if not features or not isinstance(features, dict):
            raise HTTPException(
                status_code=400, detail=" 'features' must be provided as a dictionary for what-if analysis."
            )
        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column

        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.perform_what_if, payload.get("features"))
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to perform what-if analysis: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="what-if",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.perform_what_if, payload.get("features"))
    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f" Failed to perform what-if analysis due to an unexpected internal error: {str(e)}"
        )


@app.post("/regression/analysis/feature-dependence/{feature_name}", tags=["Analysis"])
async def get_feature_dependence(
    payload: RequestPayload, feature_name: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail=" 'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail=" 'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail=" 'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")

        # ✅ Validate feature_name
        if not feature_name:
            raise HTTPException(status_code=400, detail=" 'feature_name' must be provided in the path parameter.")
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column

        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.get_feature_dependence, feature_name)
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f" Failed to generate feature dependence for '{feature_name}': {str(e)}"
            )
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="feature-dependence",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.get_feature_dependence, feature_name)
    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f" Failed to perform feature dependence analysis due to an unexpected internal error: {str(e)}",
        )


@app.post("/regression/analysis/instances", tags=["Analysis"])
async def list_instances(
    payload: RequestPayload,
    sort_by: str = "prediction",
    limit: int = 100,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user.get("user_id")
    logger.info(f"User {user_id} started regression instance analysis")
    model_service = ModelService()
    train_data_url = payload.ref_dataset
    test_data_url = payload.cur_dataset
    model_url = payload.model
    target_column = payload.target_column

    logger.info(
        f"Parameters received | train_data: {train_data_url}, test_data: {test_data_url}, "
        f"model: {model_url}, target_column: {target_column}, sort_by: {sort_by}, limit: {limit}"
    )
    try:
        model_service.load_model_and_datasets(
            model_path=model_url,
            train_data_path=train_data_url,
            test_data_path=test_data_url,
            target_column=target_column,
        )
        logger.info("Model and datasets loaded successfully.")

        # data_preview_response = handle_request(model_service.list_instances, sort_by, limit)
        # data_preview = json.loads(data_preview_response.body)
        # # Generate analysis_id
        # analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        # existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()
        # logger.info(f"Generated analysis_id: {analysis_id}")

        # if not existing:
        #     new_entry = AnalysisResult(
        #         analysis_id=analysis_id,
        #         user_id=user_id,
        #         analysis_type="regression",
        #         analysis_tab="instances",
        #         project_id="N/A",
        #         json_result=json.dumps(data_preview),
        #     )
        #     db.add(new_entry)
        #     db.commit()
        #     logger.info(f"Analysis result saved to database (analysis_id={analysis_id}).")
        # else:
        #     logger.info(f"Analysis result already exists in database (analysis_id={analysis_id}). Skipping save.")
        return handle_request(model_service.list_instances, sort_by, limit)
    except Exception as e:
        logger.error(f"Error during regression instance analysis for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/regression/analysis/dataset-comparison", tags=["Analysis"])
async def get_dataset_comparison(
    payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:

        user_id = user.get("user_id")
        model_service = ModelService()
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail=" 'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail=" 'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail=" 'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column
        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.get_dataset_comparison)
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to get dataset comparison: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="dataset-comparison",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.get_dataset_comparison)
    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f" Failed to perform dataset comparison due to an unexpected internal error: {str(e)}",
        )


# --- New enterprise feature endpoints ---
@app.post("/regression/api/features", tags=["Features"])
async def get_features_metadata(
    payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:

        user_id = user.get("user_id")
        model_service = ModelService()
        if not payload.ref_dataset:
            raise HTTPException(status_code=400, detail=" 'ref_dataset' (training dataset) is required.")
        if not payload.cur_dataset:
            raise HTTPException(status_code=400, detail=" 'cur_dataset' (test dataset) is required.")
        if not payload.model:
            raise HTTPException(status_code=400, detail=" 'model' path or URL is required.")
        if not payload.target_column:
            raise HTTPException(status_code=400, detail=" 'target_column' is required.")
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Target column '{target_column}' does not exist in the dataset."
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model or datasets: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.get_feature_metadata)
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to retrieve feature metadata: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="features",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.get_feature_metadata)
    except HTTPException:
        # Preserve custom HTTP exceptions
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f" Failed to retrieve features metadata due to an unexpected internal error: {str(e)}",
        )


@app.post("/regression/api/correlation", tags=["Features"])
async def post_correlation(
    payload: CorrelationPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:

        # Extract URL payload data from the main payload
        model_service = ModelService()
        selected: List[str] = payload.features or []
        user_id = user.get("user_id")
        train_data_url = payload.ref_dataset
        test_data_url = payload.cur_dataset
        model_url = payload.model
        target_column = payload.target_column
        missing_fields = []
        if not train_data_url:
            missing_fields.append("ref_dataset")
        if not model_url:
            missing_fields.append("model")
        if not target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(status_code=400, detail=f" Missing required fields: {', '.join(missing_fields)}")
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        # Validate required fields
        if not all([train_data_url, model_url, target_column]):
            raise HTTPException(status_code=400, detail="Missing required fields: ref_dataset, model, target_column")

        try:
            data_preview_response = handle_request(model_service.compute_correlation, selected)
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute correlation: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="correlation",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.compute_correlation, selected)
    except HTTPException:
        raise  # Preserve custom HTTP exceptions

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"⚠️ Unexpected internal error while computing correlation: {str(e)}"
        )


@app.post("/regression/api/feature-importance", tags=["Features"])
async def post_feature_importance(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()

        method = payload.get("method", "shap")
        sort_by = payload.get("sort_by", "importance")
        try:

            top_n = int(payload.get("top_n", 1000))
        except ValueError:
            raise HTTPException(status_code=400, detail=" 'top_n' must be an integer.")
        visualization = payload.get("visualization", "bar")
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )
        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column

        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:
            data_preview_response = handle_request(
                model_service.compute_feature_importance_advanced, method, sort_by, top_n, visualization
            )
            data_preview = json.loads(data_preview_response.body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute feature importance: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="feature-importance",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.compute_feature_importance_advanced, method, sort_by, top_n, visualization)
    except HTTPException:
        raise  # Preserve custom HTTP exceptions

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f" Unexpected internal error while computing feature importance: {str(e)}"
        )


@app.post("/regression/analysis/feature-interactions", tags=["Analysis"])
async def get_feature_interactions(
    url_payload: RequestPayload,
    feature1: str,
    feature2: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )
        if not feature1 or not feature2:
            raise HTTPException(status_code=400, detail=" Both 'feature1' and 'feature2' must be provided.")
        if feature1 == feature2:
            raise HTTPException(
                status_code=400, detail="'feature1' and 'feature2' must be different for interaction analysis."
            )
        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column

        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:

            data_preview_response = handle_request(model_service.get_feature_interactions, feature1, feature2)
            data_preview = json.loads(data_preview_response.body)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Invalid feature names or data: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute feature interactions: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="feature-interactions",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.get_feature_interactions, feature1, feature2)
    except HTTPException:
        raise  # preserve custom HTTP messages

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f" Unexpected internal error while performing feature interaction analysis: {str(e)}",
        )


@app.post("/regression/analysis/decision-tree", tags=["Analysis"])
async def get_decision_tree(
    url_payload: RequestPayload, user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:

        user_id = user.get("user_id")
        model_service = ModelService()
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")

        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )
        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:

            data_preview_response = handle_request(model_service.get_decision_tree)
            data_preview = json.loads(data_preview_response.body)
        except ValueError as ve:
            raise HTTPException(
                status_code=400, detail=f" Failed to generate decision tree due to invalid model or data: {str(ve)}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute decision tree visualization: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="decision-tree",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.get_decision_tree)
    except HTTPException:
        # preserve custom HTTP messages
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f" Unexpected internal error while generating decision tree analysis: {str(e)}"
        )


# --- Individual Prediction API ---
@app.post("/regression/api/individual-prediction", tags=["Prediction"])
async def post_individual_prediction(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:

        user_id = user.get("user_id")
        model_service = ModelService()
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")

        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )
        try:
            instance_idx = int(payload.get("instance_idx", 0))
            if instance_idx < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=" 'instance_idx' must be a non-negative integer.")
        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:

            data_preview_response = handle_request(model_service.individual_prediction, instance_idx)
            data_preview = json.loads(data_preview_response.body)
        except IndexError:
            raise HTTPException(
                status_code=400, detail=f" Instance index {instance_idx} is out of range for the dataset."
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=400, detail=f" Failed to perform individual prediction due to invalid input: {str(ve)}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute individual prediction: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="individual-prediction",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.individual_prediction, instance_idx)
    except HTTPException:
        # Keep custom messages
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f" Unexpected internal error while performing individual prediction: {str(e)}"
        )


# --- Regression Analysis Endpoints ---
@app.post("/regression/api/partial-dependence", tags=["Dependence"])
async def post_partial_dependence(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        feature = payload.get("feature")
        if not feature:
            raise HTTPException(status_code=400, detail=" Missing required field in request body: 'feature'.")

        # ✅ Validate num_points (must be a positive integer)
        try:
            num_points = int(payload.get("num_points", 20))
            if num_points <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=" 'num_points' must be a positive integer.")

        if not feature:
            raise HTTPException(status_code=400, detail="Missing 'feature'")
        try:
            num_points = int(payload.get("num_points", 20))
            if num_points <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=" 'num_points' must be a positive integer.")
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")

        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.partial_dependence, feature, num_points)
            data_preview = json.loads(data_preview_response.body)
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Feature '{feature}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(
                status_code=400,
                detail=f" Failed to compute partial dependence due to invalid parameters or data: {str(ve)}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute partial dependence: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="partial-dependence",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.partial_dependence, feature, num_points)
    except HTTPException:
        # keep custom messages
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f" Unexpected internal error while performing partial dependence analysis: {str(e)}"
        )


@app.post("/regression/api/shap-dependence", tags=["Dependence"])
async def post_shap_dependence(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        feature = payload.get("feature")
        color_by = payload.get("color_by")
        if not feature:
            raise HTTPException(status_code=400, detail="Missing required field in request body: 'feature'.")
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:

            data_preview_response = handle_request(model_service.shap_dependence, feature, color_by)
            data_preview = json.loads(data_preview_response.body)
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f" Feature '{feature}' or color_by '{color_by}' not found in dataset."
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=400,
                detail=f" Failed to compute SHAP dependence due to invalid parameters or data: {str(ve)}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute SHAP dependence: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="shap-dependence",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.shap_dependence, feature, color_by)
    except HTTPException:
        # Preserve custom messages
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Unexpected internal error while performing SHAP dependence analysis: {str(e)}"
        )


@app.post("/regression/api/ice-plot", tags=["Dependence"])
async def post_ice_plot(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        feature = payload.get("feature")
        if not feature:
            raise HTTPException(status_code=400, detail=" Missing required field in request body: 'feature'.")
        num_points = int(payload.get("num_points", 20))
        num_instances = int(payload.get("num_instances", 20))
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.ice_plot, feature, num_points, num_instances)
            data_preview = json.loads(data_preview_response.body)
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Feature '{feature}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(
                status_code=400, detail=f"Failed to compute ICE plot due to invalid parameters or data: {str(ve)}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute ICE plot: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="ice-plot",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.ice_plot, feature, num_points, num_instances)

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f" Unexpected internal error while performing ICE plot analysis: {str(e)}"
        )


# --- Section 5 APIs ---
@app.post("/regression/api/interaction-network", tags=["Interactions"])
async def post_interaction_network(
    url_payload: RequestPayload,
    payload: Dict = Body({}),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        top_k = int(payload.get("top_k", 30))
        sample_rows = int(payload.get("sample_rows", 200))
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:

            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded properly: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.interaction_network, top_k, sample_rows)
            data_preview = json.loads(data_preview_response.body)
        except KeyError as ke:
            raise HTTPException(status_code=400, detail=f"Column not found in dataset: {str(ke)}")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Failed to compute interaction network: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Failed to compute interaction network: {str(e)}")
        # Generate analysis_id
        analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()

        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="interaction-network",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.interaction_network, top_k, sample_rows)
    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f" Unexpected internal error while performing interaction network analysis: {str(e)}",
        )


@app.post("/regression/api/pairwise-analysis", tags=["Interactions"])
async def post_pairwise_analysis(
    url_payload: RequestPayload,
    payload: Dict = Body(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        user_id = user.get("user_id")
        model_service = ModelService()
        f1 = payload.get("feature1")
        f2 = payload.get("feature2")
        if not f1 or not f2:
            raise HTTPException(status_code=400, detail=" Missing 'feature1' or 'feature2' in payload")
        color_by = payload.get("color_by")
        sample_size = int(payload.get("sample_size", 1000))
        missing_fields = []
        if not url_payload.ref_dataset:
            missing_fields.append("ref_dataset")
        if not url_payload.model:
            missing_fields.append("model")
        if not url_payload.target_column:
            missing_fields.append("target_column")
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f" Missing required fields in URL payload: {', '.join(missing_fields)}"
            )

        train_data_url = url_payload.ref_dataset
        test_data_url = url_payload.cur_dataset
        model_url = url_payload.model
        target_column = url_payload.target_column
        try:
            model_service.load_model_and_datasets(
                model_path=model_url,
                train_data_path=train_data_url,
                test_data_path=test_data_url,
                target_column=target_column,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=" Model file or dataset not found at the specified path.")
        except KeyError:
            raise HTTPException(status_code=400, detail=f" Target column '{target_column}' not found in dataset.")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" Dataset could not be loaded: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error while loading model/dataset: {str(e)}")
        try:
            data_preview_response = handle_request(model_service.pairwise_analysis, f1, f2, color_by, sample_size)
        except Exception as e:
            raise HTTPException(
                status_code=502, detail=f"Error fetching pairwise analysis from model service: {str(e)}"
            )
        try:
            data_preview = json.loads(data_preview_response.body)
        except (ValueError, TypeError, json.JSONDecodeError):
            raise HTTPException(
                status_code=500, detail="Invalid or malformed JSON response received from model service."
            )
        # Generate analysis_id
        try:
            analysis_id = generate_analysis_id(user_id=user_id, data_preview=data_preview)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating analysis ID: {str(e)}")
        try:
            existing = db.query(AnalysisResult).filter_by(analysis_id=analysis_id).first()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database query error while checking existing pairwise analysis record: {str(e)}",
            )
        if not existing:
            new_entry = AnalysisResult(
                analysis_id=analysis_id,
                user_id=user_id,
                analysis_type="regression",
                analysis_tab="pairwise-analysis",
                project_id="N/A",
                json_result=json.dumps(data_preview),
            )
            db.add(new_entry)
            db.commit()
        return handle_request(model_service.pairwise_analysis, f1, f2, color_by, sample_size)
    except HTTPException:
        raise

    # ✅ Catch-all for unexpected errors
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error occurred while generating pairwise analysis: {str(e)}"
        )


# --- AI Explanation Endpoint ---
@app.post("/regression/analysis/explain-with-ai", tags=["AI Analysis"])
async def explain_with_ai(payload: Dict = Body(...), user: str = Depends(get_current_user)):
    """
    Generate an AI-powered explanation of the current analysis results.

    Expected payload:
    {
        "analysis_type": "overview|feature_importance|classification_stats|...",
        "analysis_data": {...}  # The data to be explained
    }
    """
    try:
        if not user or not user.get("user_id"):
            raise HTTPException(status_code=401, detail="User authentication failed or user ID missing.")
        analysis_type = payload.get("analysis_type")
        analysis_data = payload.get("analysis_data", {})
        ai_explanation_service = AIExplanationService()
        if not analysis_type:
            raise HTTPException(
                status_code=400,
                detail=" Missing 'analysis_type' in payload. Example: 'overview' or 'feature_importance'.",
            )
        if not isinstance(analysis_data, dict):
            raise HTTPException(
                status_code=400, detail=" 'analysis_data' must be a dictionary containing the data to explain."
            )
        ai_explanation_service = AIExplanationService()
        try:

            # Generate AI explanation
            explanation = ai_explanation_service.generate_explanation(analysis_data, analysis_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f" AI explanation could not be generated: {str(ve)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f" Unexpected error during AI explanation: {str(e)}")
        return JSONResponse(status_code=200, content=explanation)

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"    Failed to generate AI explanation due to an internal error: {str(e)}"
        )


app.include_router(regression.routers)
