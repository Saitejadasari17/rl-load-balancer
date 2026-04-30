# 🚀 Adaptive Load Balancing with Reinforcement Learning

**A full-stack capstone project combining Deep RL with a modern web interface**

![Status](https://img.shields.io/badge/Status-Complete-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![React](https://img.shields.io/badge/React-18.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

---

## 📋 Project Overview

An **Adaptive Load Balancer** trained using **Deep Reinforcement Learning** on **real Microsoft Azure Functions dataset**. The RL agent learns optimal load distribution strategies by training on actual production traces.

### ✨ Key Features
- ✅ **Real Data Training**: Trained on 10,000+ Azure function invocation traces
- ✅ **Advanced RL Algorithm**: PPO (Proximal Policy Optimization) implementation
- ✅ **Baseline Comparison**: Round Robin, Least Connections, Random, Weighted RR
- ✅ **Full Web Dashboard**: Upload datasets, visualize training, view results
- ✅ **Production Ready**: Deployed on Vercel + Render (free tier)
- ✅ **Comprehensive Metrics**: Latency, utilization, fairness analysis

---

## 🎯 Research Results

Trained on **real Microsoft Azure Functions dataset (2019)**:

| Algorithm | Avg Latency | Utilization | Notes |
|-----------|------------|-------------|-------|
| Round Robin | 127.26 ms | 35.6% | Baseline: simple |
| Least Connections | **117.61 ms** | 38.6% | Best Baseline |
| Random | 162.24 ms | 43.8% | Worst performer |
| RL Agent (PPO) | 124.59 ms | 37.4% | Learned policy |

**Fairness Index**: 0.927 (higher = more balanced)

---

## 🌐 Live Demo

| Component | URL | Status |
|-----------|-----|--------|
| **Frontend** | https://rl-load-balancer.vercel.app | 🟢 Live |
| **API Docs** | https://rl-load-balancer-api.onrender.com/docs | 🟢 Live |
| **GitHub** | https://github.com/YOUR_USERNAME/rl-load-balancer | - |

---

## 🛠 Tech Stack

### Backend (Python)
```
FastAPI (Server)
  ├── Stable-Baselines3 (PPO RL)
  ├── Gymnasium (Environment)
  ├── PyTorch (Neural Network)
  ├── Pandas & NumPy (Data)
  └── [Your RL Code]
```

### Frontend (React)
```
React 18.2 (UI)
  ├── TailwindCSS (Styling)
  ├── Recharts (Charts)
  └── Lucide Icons (UI)
```

### Deployment
```
Vercel (Frontend)    ↔  Render (Backend)
Free tier            ↔  Free tier
Auto HTTPS           ↔  Python 3.12
Global CDN           ↔  Auto restart
```

---

## 📁 Project Structure

```
rl-load-balancer/
│
├── backend/                        ← FastAPI Server
│   ├── main.py                     ← REST API endpoints
│   ├── requirements.txt
│   └── Dockerfile (optional)
│
├── frontend/                       ← React Dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── UploadSection.jsx   ← File upload
│   │   │   ├── TrainingProgress.jsx ← Training UI
│   │   │   ├── MetricCards.jsx     ← Results summary
│   │   │   └── ResultCharts.jsx    ← Visualizations
│   │   └── index.css
│   ├── package.json
│   └── tailwind.config.js
│
├── src/                            ← Original RL Code
│   ├── environment.py              ← LoadBalancerEnv
│   ├── agent.py                    ← RL training/evaluation
│   ├── baselines.py                ← Baseline algorithms
│   ├── azure_dataset.py            ← Dataset loader
│   └── extract_compressed.py       ← .tar.xz support
│
├── data/                           ← Datasets
│   ├── azurefunctions-dataset2019.tar.xz
│   └── extracted/                  ← Extracted CSVs
│
├── models/                         ← Trained Models
│   └── rl_agent_azure.zip
│
├── results/                        ← Training Results
│   ├── api_training_results.json
│   └── *.png (charts)
│
├── DEPLOYMENT.md                   ← Full deployment guide
├── QUICKSTART.md                   ← Local setup guide
├── vercel.json                     ← Vercel config
├── render.yaml                     ← Render config
└── README.md (this file)
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.12+
- Node.js 16+
- Git

### 1️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

✅ Backend runs at: http://localhost:8000  
📖 API Docs: http://localhost:8000/docs

### 2️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start
```

✅ Frontend runs at: http://localhost:3000

### 3️⃣ Start Training

1. Open http://localhost:3000
2. Upload a CSV dataset
3. Click "Start Training"
4. Watch results in real-time!

---

## 🧬 RL Algorithm Details

### Environment
- **State**: 9-dimensional vector [cpu₁, queue₁, latency₁, cpu₂, queue₂, latency₂, cpu₃, queue₃, latency₃]
- **Action**: Discrete(3) - choose server 0, 1, or 2
- **Reward**: -latency (lower latency = higher reward)
- **Data Source**: Real Azure function invocation traces

### Training
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Neural Network**: MLP Policy with 64 hidden units
- **Timesteps**: 50,000 (default)
- **Learning Rate**: 3e-4
- **Batch Size**: 64

### Baselines
1. **Round Robin** - Cycle through servers
2. **Least Connections** - Route to server with fewest requests
3. **Random** - Random server selection
4. **Weighted Round Robin** - Proportional to server capacity

---

## 📊 Using the Dashboard

### Upload Section
- Drag & drop CSV files
- Supports: Function traces, request logs, performance data
- Auto-detects numeric columns

### Configuration
- **Timesteps**: Training duration (default: 50,000)
- **Episodes**: Evaluation runs (default: 10)
- **Servers**: Number of servers to simulate (default: 3)

### Results View
- **Latency Comparison**: Bar chart comparing algorithms
- **Utilization**: Server load balance chart
- **Metrics**: Improvement %, fairness index
- **Baseline Details**: Each algorithm's performance

---

## 🚢 Deployment to Production

### Frontend (Vercel)
```bash
git push origin main
# Vercel auto-deploys on GitHub push
```

### Backend (Render)
```bash
# Just push to GitHub
# Render auto-deploys based on render.yaml
```

**Full guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📈 API Endpoints

### Health & Status
```
GET  /                    # Health check
GET  /training-status     # Current training state
GET  /metrics            # System capabilities
```

### Training
```
POST /train              # Start training
GET  /results            # Last training results
POST /upload-dataset     # Upload CSV file
```

### Interactive Docs
```
http://localhost:8000/docs                  # Swagger UI
http://localhost:8000/redoc                 # ReDoc
```

---

## 🔬 Research Methodology

1. **Data Collection**: Real Azure Functions dataset (50,000+ functions)
2. **Preprocessing**: Normalized to [0.05, 0.3] load range
3. **Baseline Evaluation**: 4 static algorithms tested
4. **RL Training**: PPO trained for 50,000 timesteps
5. **Evaluation**: 10 episodes × 500 steps per episode
6. **Metrics**: Latency, utilization, fairness, improvement %

---

## 💡 Key Insights

✅ **What Works Well**:
- RL agent learns to balance loads dynamically
- Fairness improves server utilization
- PPO converges quickly on load balancing task
- Real data training shows practical applicability

⚠️ **Challenges**:
- Limited training data (10K samples) requires hyperparameter tuning
- Free tier deployments have resource constraints
- Cold starts on Render add ~30s latency

📊 **Future Improvements**:
- Multi-objective optimization (latency + throughput)
- Distributed training on GPU
- Real-time dataset streaming
- Integration with actual load balancers (Nginx, HAProxy)

---

## 📚 Papers & References

- PPO Algorithm: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- Deep RL for Networking: [Mao et al., 2016](https://arxiv.org/abs/1603.05061)
- Azure Functions Dataset: [Shahrad et al., 2020](https://arxiv.org/abs/1902.03356)

---

## 👨‍💻 Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│              User (Web Browser)                      │
│          https://rl-load-balancer.vercel.app        │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  React Frontend  │ (Vercel)
        │  - Upload UI     │
        │  - Charts        │
        │  - Results       │
        └────────┬────────┘
                 │ HTTPS REST API
        ┌────────▼──────────────────┐
        │  FastAPI Backend          │ (Render)
        │  - Training Orchestrator  │
        │  - Baseline Evaluation    │
        │  - RL Training            │
        │  - Results Aggregation    │
        └────────┬──────────────────┘
                 │
        ┌────────▼──────────────────┐
        │  RL Components            │
        │  - LoadBalancerEnv        │
        │  - PPO Agent              │
        │  - Baseline Algorithms    │
        └─────────────────────────────┘
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add DDPG/TD3 algorithms
- [ ] Support multi-objective optimization
- [ ] Real Kubernetes integration
- [ ] Performance profiling
- [ ] Mobile app version

---

## 📄 License

MIT License - See LICENSE file

---

## 🎓 Capstone Project

**School**: [Your University]  
**Course**: Capstone Project / Senior Design  
**Term**: Spring 2026  
**Advisor**: [Your Advisor]  

---

## 📞 Questions?

- 📧 Email: your.email@university.edu
- 🐙 GitHub Issues: https://github.com/YOUR_USERNAME/rl-load-balancer/issues
- 💬 Discussions: https://github.com/YOUR_USERNAME/rl-load-balancer/discussions

---

**Made with ❤️ for the future of adaptive systems**
