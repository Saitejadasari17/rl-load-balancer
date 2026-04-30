# 🎯 Quick Start Guide - Run Locally

## ⚡ 5-Minute Local Setup

### Terminal 1: Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
✅ Backend running at: http://localhost:8000
📖 API docs at: http://localhost:8000/docs

### Terminal 2: Frontend
```bash
cd frontend
npm install
npm start
```
✅ Frontend running at: http://localhost:3000

---

## 📱 Using the App

1. **Open Browser**: http://localhost:3000
2. **Upload CSV File**: Drag and drop a dataset
3. **Configure Training**:
   - Timesteps: 50000 (default)
   - Episodes: 10 (default)
   - Servers: 3 (default)
4. **Start Training**: Click "Start Training"
5. **View Results**: Charts and metrics display automatically

---

## 🧪 API Testing

### Using Swagger UI
```
http://localhost:8000/docs
```
- Click any endpoint
- Enter parameters
- Click "Try it out"
- See response

### Using curl
```bash
# Health check
curl http://localhost:8000/

# Get metrics
curl http://localhost:8000/metrics

# Training status
curl http://localhost:8000/training-status
```

---

## 🚀 Next Steps

After local testing works:
1. Push to GitHub
2. Deploy frontend to Vercel (DEPLOYMENT.md)
3. Deploy backend to Render (DEPLOYMENT.md)
4. Share live URL!

---

## 📦 Project Structure
```
.
├── backend/
│   ├── main.py           ← FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/   ← React components
│   └── package.json
├── src/                  ← Original RL code
│   ├── environment.py
│   ├── agent.py
│   └── baselines.py
├── DEPLOYMENT.md         ← Full deployment guide
└── README.md
```

---

**Troubleshooting**: See DEPLOYMENT.md for common issues
