# Prophecy AI - Real Estate Intelligence Platform

> MSc Data Science Project | Dubai Real Estate Demand Prediction using LSTM + XGBoost

A real-time streaming pipeline that predicts rent demand and sale prices for Dubai properties using Apache Kafka, Spark, TensorFlow (LSTM), and XGBoost. Fronted by a Streamlit dashboard with dark-mode UI.

---

## Architecture Overview

```
┌─────────────┐     ┌─────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────┐
│   APIs /    │────▶│  Kafka  │────▶│Spark Stream │────▶│  LSTM    │────▶│ Streamlit│
│ Sample Data │     │ Broker  │     │ Processor   │     │ XGBoost  │     │ Dashboard│
└─────────────┘     └─────────┘     └─────────────┘     └──────────┘     └──────────┘
```

**Tech Stack:** Python 3.10, Kafka, Spark 3.5, TensorFlow 2.15, XGBoost 2.0, Streamlit 1.29, Docker

---

## Prerequisites

### Option A: Docker (Recommended)

1. **Install Docker Desktop**
   - Windows/Mac: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Linux: `sudo apt update && sudo apt install docker.io docker-compose`
   - Verify: `docker --version` and `docker-compose --version`

2. **Install Git**
   - [https://git-scm.com/downloads](https://git-scm.com/downloads)
   - Verify: `git --version`

3. **(Optional) Python 3.10+** - only if running locally without Docker
   - [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Verify: `python --version`

### Option B: Free Cloud (No Local Install)

Skip to [Free Cloud Deployment](#free-cloud-deployment) section below.

---

## Quick Start (Docker - 5 minutes)

### Step 1: Clone / Download Project

```bash
git clone <repo-url> prophecy-ai
cd prophecy-ai
```

Or extract the ZIP file and `cd` into the folder.

### Step 2: Start Infrastructure

```bash
docker-compose up -d zookeeper kafka spark-master spark-worker
```

Wait ~30 seconds for services to initialize.

### Step 3: Train ML Models

```bash
# Build and run training container
docker-compose up --build ml-models
```

This generates:
- `models/lstm_*.keras` - LSTM models per area
- `models/xgboost_model.json` - XGBoost classifier
- `models/*.json` - Performance metrics

### Step 4: Start Data Stream + Dashboard

```bash
# Terminal 1: Start property data generator
docker-compose up data-generator

# Terminal 2: Start Spark processor (optional - for streaming features)
docker-compose up spark-processor

# Terminal 3: Start Streamlit dashboard
docker-compose up streamlit
```

### Step 5: Open Dashboard

Navigate to: **http://localhost:8501**

You should see the Prophecy AI dashboard with:
- Live property listings
- AI prediction panel
- Market analytics
- Demand forecasts

---

## Local Development (No Docker)

### Step 1: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn tensorflow xgboost streamlit kafka-python pyspark

# Or install from requirements files
pip install -r ml_models/requirements.txt
pip install -r streamlit_app/requirements.txt
```

### Step 3: Train Models

```bash
python train_local.py
```

### Step 4: Start Streamlit

```bash
streamlit run streamlit_app/app.py
```

Open **http://localhost:8501**

---

## Free Cloud Deployment

### Option 1: Render (Recommended - Free Tier)

Render offers free web services with 512MB RAM (enough for Streamlit).

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/prophecy-ai.git
   git push -u origin main
   ```

2. **Create Render Account**
   - Go to [https://render.com](https://render.com)
   - Sign up with GitHub

3. **Deploy Streamlit Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Settings:
     - **Name:** `prophecy-ai`
     - **Runtime:** `Python 3`
     - **Build Command:** `pip install -r streamlit_app/requirements.txt && python train_local.py`
     - **Start Command:** `streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0`
     - **Plan:** Free
   - Click "Create Web Service"

4. **Wait for Build** (~5-10 minutes)
   - Render will install deps, train models, and start the app
   - URL will be: `https://prophecy-ai.onrender.com`

5. **Validation**
   - Open the URL
   - Check Dashboard loads with sample data
   - Go to "AI Prediction" page, enter property details, click "Run Prediction"
   - Verify fair price and demand scores appear

### Option 2: Railway (Free Tier - $5 credit/month)

1. Go to [https://railway.app](https://railway.app)
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo
4. Add a service, set start command:
   ```bash
   pip install -r streamlit_app/requirements.txt && python train_local.py && streamlit run streamlit_app/app.py --server.port=$PORT
   ```
5. Deploy and get public URL

### Option 3: GitHub Codespaces (Free - 60 hrs/month)

1. Push code to GitHub repo
2. Go to repo → Code → Codespaces → Create codespace on main
3. In terminal:
   ```bash
   pip install -r streamlit_app/requirements.txt
   python train_local.py
   streamlit run streamlit_app/app.py
   ```
4. Click "Open in Browser" when popup appears
5. Share URL with others (it is public while codespace runs)

### Option 4: AWS Free Tier (EC2 - 750 hrs/month for 12 months)

1. **Create AWS Account**: [https://aws.amazon.com/free](https://aws.amazon.com/free)
2. **Launch EC2 Instance**:
   - Search "EC2" → Launch Instance
   - Name: `prophecy-ai`
   - AMI: Ubuntu Server 22.04 LTS (Free tier eligible)
   - Instance type: `t2.micro` (Free tier)
   - Key pair: Create new (download .pem file)
   - Network: Allow HTTP (port 80) and Custom TCP (port 8501)
   - Storage: 20 GB
   - Launch

3. **Connect to Instance**:
   ```bash
   chmod 400 your-key.pem
   ssh -i your-key.pem ubuntu@YOUR_EC2_IP
   ```

4. **Setup Environment**:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv git
   git clone https://github.com/YOUR_USERNAME/prophecy-ai.git
   cd prophecy-ai
   python3 -m venv venv
   source venv/bin/activate
   pip install -r streamlit_app/requirements.txt
   python train_local.py
   ```

5. **Run with NoHup** (keeps running after disconnect):
   ```bash
   nohup streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0 > app.log 2>&1 &
   ```

6. **Access**: `http://YOUR_EC2_IP:8501`

7. **(Optional) Add Domain**:
   - Buy domain on Namecheap (~$10/year)
   - Point A record to EC2 IP
   - Install Nginx as reverse proxy

### Option 5: Google Cloud Run (Free Tier - 2M requests/month)

1. **Install GCloud CLI**: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
2. **Create Project**:
   ```bash
   gcloud projects create prophecy-ai --name="Prophecy AI"
   gcloud config set project prophecy-ai
   ```
3. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
   ```
4. **Submit Build**:
   ```bash
   gcloud builds submit --tag gcr.io/prophecy-ai/streamlit-app ./streamlit_app
   ```
5. **Deploy**:
   ```bash
   gcloud run deploy prophecy-ai      --image gcr.io/prophecy-ai/streamlit-app      --platform managed      --region us-central1      --allow-unauthenticated      --port 8501
   ```
6. **Get URL** from output and open in browser

---

## Validation Checklist

After deployment, verify these features:

| Feature | How to Test | Expected Result |
|---------|-------------|-----------------|
| Dashboard Load | Open main URL | See 4 metric cards, quick actions, activity feed |
| Property Search | Go to "Property Search", select filters | Grid of property cards with prices and demand badges |
| AI Prediction | Go to "AI Prediction", fill form, click Run | Fair price estimate, rent/sale demand scores, model insights |
| Analytics | Go to "Analytics" | Line chart for rent trends, bar chart for community demand |
| Saved Properties | Go to "Saved" | Sample saved properties with tracking badges |
| Dark Theme | Check UI appearance | Dark navy background (#0f172a) with purple accents |
| Responsive | Resize browser window | Cards stack on mobile, grid on desktop |

### Sample Test Case

1. Navigate to **AI Prediction**
2. Enter:
   - Location: `Dubai Marina`
   - Type: `Apartment`
   - Bedrooms: `3`
   - Bathrooms: `3`
   - Size: `1850` sqft
   - Floor: `22`
   - Price: `2100000`
   - DOM: `12`
3. Click **Run Prediction**
4. Verify output contains:
   - Fair Market Price (AED 1.6M - 2.8M range)
   - Demand classification (High/Very High/Moderate)
   - Confidence indicator
   - Model insight text

---

## Project Structure

```
prophecy-ai/
├── docker-compose.yml          # Orchestrates all services
├── train_local.py              # Local model training script
├── .env.example                # Environment variables template
├── data/
│   └── sample_properties.csv   # 2000 synthetic Dubai listings
├── models/                     # Generated model files (after training)
├── data_generator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── producer.py             # Kafka producer simulating live listings
├── spark_processor/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── stream_processor.py     # PySpark streaming + feature engineering
├── ml_models/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train_models.py         # Orchestrates LSTM + XGB training
│   ├── lstm_model.py           # LSTM time-series forecaster
│   └── xgboost_model.py        # XGBoost demand classifier
└── streamlit_app/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                  # Dashboard UI (dark theme)
```

---

## Troubleshooting

### Issue: Kafka connection refused
**Fix:** Wait 30s after `docker-compose up` for Kafka to initialize. Or increase sleep in `producer.py`.

### Issue: `train_local.py` fails with TensorFlow errors
**Fix:** Ensure Python 3.10 (not 3.12). TF 2.15 does not support Python 3.12.
```bash
python3.10 -m venv venv
```

### Issue: Streamlit shows "No models found"
**Fix:** Run training first:
```bash
python train_local.py
```
Models must exist in `models/` before starting Streamlit.

### Issue: Docker build takes too long
**Fix:** Pre-download base images:
```bash
docker pull python:3.10-slim
docker pull bitnami/spark:3.5.0
docker pull confluentinc/cp-kafka:7.5.0
```

### Issue: Out of memory on free tier
**Fix:** Reduce sample data size in `data/sample_properties.csv` to 500 rows. Or deploy only Streamlit + pre-trained models without Kafka/Spark.

### Issue: Render build timeout
**Fix:** Training takes ~5-10 min. Add a `render.yaml` with longer timeout, or train locally and commit `models/` folder to git.

---

## Data Sources

This project uses **synthetic data** generated to mimic Dubai real estate patterns:
- Areas: Dubai Marina, Downtown, Palm Jumeirah, JLT, Arabian Ranches, Bluewaters
- Price ranges based on 2024-2025 market reports
- Seasonal factors (summer dip) applied

For production, replace `producer.py` with real API feeds from Bayut or Property Finder.

---

## License

Academic use only. Part of MSc Data Science dissertation project.

## Contact

research@proppredict.ai | MDX Dubai Campus
