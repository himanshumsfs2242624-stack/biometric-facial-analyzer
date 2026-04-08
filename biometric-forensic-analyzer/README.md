#  Biometric Forensic Analyzer

A scalable AI-powered system for **biometric facial analysis and digital forensics** built using FastAPI, Celery, and Docker.

---

##  Overview

The **Biometric Forensic Analyzer** processes video inputs to extract and analyze facial biometrics and detect forensic inconsistencies such as tampering or deepfake artifacts.

It combines:

*  Computer Vision (Face Mesh, Eye Tracking, YOLO)
*  Digital Forensics (ELA, PRNU, Metadata)
*  Scalable Backend (FastAPI + Celery + Docker)

---

## Architecture

* **FastAPI** → REST API layer
* **Celery** → Asynchronous background processing
* **Redis / Queue** → Task broker
* **Docker Compose** → Multi-service orchestration

 Entire system runs using **containerized microservices** ([testdriven.io][1])

---

## Project Structure

```
biometric-forensic-analyzer/
├── docker/
├── models/
├── src/
│   ├── api/
│   ├── biometrics/
│   ├── forensics/
│   ├── matching/
│   ├── core/
│   └── utils/
├── tests/
├── data/
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Features

### 🔹 Biometric Analysis

* 468-point face mesh tracking
* Eye tracking, blinking, gaze detection
* Mouth and lip movement analysis
* Custom tongue detection (YOLO)

### 🔹 Forensic Analysis

* Metadata extraction (EXIF)
* Error Level Analysis (ELA)
* PRNU (sensor noise fingerprinting)

### 🔹 System Features

* Async video processing (Celery workers)
* Modular architecture
* Dockerized deployment
* Automated testing (PyTest)

---

##  Installation & Setup

### 🔥 Option 1: Run with Docker (Recommended)

> Requires: Docker + Docker Compose

```bash
git clone https://github.com/himanshumsfs2242624-stack/biometric-facial-analyzer.git
cd biometric-facial-analyzer/biometric-forensic-analyzer

docker-compose up --build
```

👉 This starts:

* API server
* Worker service
* Background queue

(Similar setup is standard in FastAPI + Celery systems ([GitHub][2]))

---

### ⚡ Option 2: Manual Setup (Without Docker)

#### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run FastAPI server

```bash
uvicorn src.main:app --reload
```

#### 4. Start Celery worker

```bash
celery -A src.core.celery_app worker --loglevel=info
```

---

## 🌐 API Usage

### 🔹 Open API Docs

```
http://localhost:8000/docs
```

---

### 🔹 Example: Upload Video

```bash
curl -X POST "http://localhost:8000/upload" \
-F "file=@sample.mp4"
```

---

### 🔹 Get Results

```bash
curl http://localhost:8000/report/{job_id}
```

---

## 📊 Output

Generated outputs include:

* JSON reports (biometric + forensic)
* Annotated video with overlays
* Frame-level analysis

Stored in:

```
data/output_reports/
```

---

## 📸 Demo Structure (Add This)

```
demo/
├── input_sample.mp4
├── output_report.json
├── output_video.mp4
└── screenshots/
```

👉 Add:

* Before/After frames
* API screenshots
* Swagger UI images

---

## ☁️ Deployment

### 🔹 Option 1: Docker (Production)

```bash
docker-compose -f docker-compose.yml up -d --build
```

---

### 🔹 Option 2: Cloud Deployment

#### Deploy on:

* AWS EC2
* Render
* DigitalOcean

Steps:

1. Push repo to GitHub
2. Install Docker on server
3. Run:

```bash
docker-compose up -d
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## ⚠️ Notes

* Large ML models are stored in `/models/`
* Consider using external storage (S3 / Drive) for scalability
* Ensure FFmpeg is installed for frame extraction

---

## 🔮 Future Improvements

* Frontend dashboard (React / Streamlit)
* Real-time analysis
* Model optimization
* GPU acceleration

---

## 👨‍💻 Author

Himanshu Baberwal

---

## ⭐ Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

MIT License

[1]: https://testdriven.io/courses/fastapi-celery/docker/?utm_source=chatgpt.com "The Definitive Guide to Celery and FastAPI - Dockerizing Celery and FastAPI | TestDriven.io"
[2]: https://github.com/testdrivenio/fastapi-celery?utm_source=chatgpt.com "GitHub - testdrivenio/fastapi-celery: Example of how to handle background processes with FastAPI, Celery, and Docker"
