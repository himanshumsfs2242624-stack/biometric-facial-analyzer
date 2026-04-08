# src/api/endpoints_video.py

import os
import uuid
from fastapi import APIRouter, UploadFile
from app.services.processing_service import process_video
# Import Celery task
from src.worker.tasks import process_video_task

router = APIRouter()

UPLOAD_DIR = "data/raw_uploads"


@router.post("/upload")
async def upload_video(file: UploadFile):
    result = await process_video(file)
    return {"job_id": result}
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Save file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Trigger Celery task
        task = process_video_task.delay(file_path)

        return {
            "task_id": task.id,
            "status": "processing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))