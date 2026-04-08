# src/main.py

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from src.api.endpoints_video import router as video_router
from src.api.endpoints_report import router as report_router

# Initialize FastAPI app
app = FastAPI(
    title="Biometric & Forensic Video Analysis API",
    version="1.0.0"
)

# Configure CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router, prefix="/video", tags=["Video"])
app.include_router(report_router, prefix="/report", tags=["Report"])

# Create required directories at startup
REQUIRED_DIRS = [
    "data/raw_uploads",
    "data/extracted_frames",
    "data/output_reports"
]


@app.on_event("startup")
async def startup_event():
    for directory in REQUIRED_DIRS:
        os.makedirs(directory, exist_ok=True)


# Root endpoint (optional but useful)
@app.get("/")
def root():
    return {"message": "API is running"}


# For direct execution (optional, useful outside Docker)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)