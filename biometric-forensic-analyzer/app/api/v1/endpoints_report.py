# src/api/endpoints_report.py

import os
import json
from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

router = APIRouter()

REPORTS_DIR = "data/output_reports"


@router.get("/report/{task_id}")
async def get_report(task_id: str):
    try:
        # Get Celery task result
        task_result = AsyncResult(task_id)

        # If task is still pending or processing
        if task_result.state in ["PENDING", "STARTED", "RETRY"]:
            return {
                "task_id": task_id,
                "status": task_result.state.lower()
            }

        # If task failed
        if task_result.state == "FAILURE":
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(task_result.result)
            }

        # If task completed successfully
        if task_result.state == "SUCCESS":
            report_path = os.path.join(REPORTS_DIR, f"{task_id}.json")

            if not os.path.exists(report_path):
                raise HTTPException(
                    status_code=404,
                    detail="Report not found"
                )

            with open(report_path, "r") as f:
                report_data = json.load(f)

            return {
                "task_id": task_id,
                "status": "completed",
                "report": report_data
            }

        # Fallback (unknown state)
        return {
            "task_id": task_id,
            "status": task_result.state.lower()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))