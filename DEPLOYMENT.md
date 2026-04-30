# 🚀 Deployment Guide - RL Load Balancer Web App

## 📋 Pre-Deployment Checklist

- [ ] GitHub repository created and code pushed
- [ ] Vercel account created (free)
- [ ] Render account created (free)
- [ ] Backend runs locally: `cd backend && python -m uvicorn main:app --reload`
- [ ] Frontend runs locally: `cd frontend && npm start`

---

## 🔧 Local Development Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
# Opens at http://localhost:8000
# API docs: http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm start
# Opens at http://localhost:3000
# Set REACT_APP_API_URL=http://localhost:8000 in .env
```

---

## 🌍 Deploy to Vercel (Frontend)

### Step 1: Push code to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rl-load-balancer.git
git push -u origin main
```

### Step 2: Deploy Frontend
1. Go to [vercel.com](https://vercel.com)
2. Click **"New Project"**
3. Import your GitHub repo
4. **Framework**: React
5. **Root Directory**: ./frontend
6. **Build Command**: `npm run build`
7. **Install Command**: `npm install`
8. Add **Environment Variable**:
   - `REACT_APP_API_URL` = `https://rl-load-balancer-api.onrender.com` (from Step 3)
9. Click **Deploy**

✅ Your frontend is now live! You'll get a URL like: `https://rl-load-balancer.vercel.app`

---

## 🌍 Deploy to Render (Backend)

### Step 1: Create Web Service
1. Go to [render.com](https://render.com)
2. Click **"New+"** → **"Web Service"**
3. Select your GitHub repository
4. **Name**: `rl-load-balancer-api`
5. **Environment**: Python 3
6. **Build Command**: 
   ```
   pip install -r backend/requirements.txt
   ```
7. **Start Command**:
   ```
   cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
8. **Plan**: Free
9. Click **"Create Web Service"**

✅ Your backend is now deployed! You'll get a URL like: `https://rl-load-balancer-api.onrender.com`

### Step 2: Update Frontend Environment
1. Go back to Vercel dashboard
2. Select your frontend project
3. **Settings** → **Environment Variables**
4. Update `REACT_APP_API_URL` to your Render backend URL
5. Redeploy (Vercel auto-redeploys on GitHub push)

---

## 🔐 Production URLs

After deployment, you'll have:

| Component | URL | Provider |
|-----------|-----|----------|
| **Frontend** | https://rl-load-balancer.vercel.app | Vercel |
| **API Docs** | https://rl-load-balancer-api.onrender.com/docs | Render |
| **API Health** | https://rl-load-balancer-api.onrender.com/ | Render |

---

## 🧪 Testing After Deployment

### Test Backend API
```bash
# Health check
curl https://rl-load-balancer-api.onrender.com/

# Get metrics
curl https://rl-load-balancer-api.onrender.com/metrics

# View API docs (interactive)
# https://rl-load-balancer-api.onrender.com/docs
```

### Test Frontend
1. Visit `https://rl-load-balancer.vercel.app`
2. Should see the dashboard
3. Click "Start Training" to begin
4. Check backend logs at render.com dashboard

---

## 📊 Monitoring & Logs

### Frontend (Vercel)
- Dashboard: https://vercel.com/dashboard
- Check deployments, build logs, analytics
- Free automatic HTTPS and CDN

### Backend (Render)
- Dashboard: https://render.com/dashboard
- Check service health, logs, resource usage
- Free auto-restarts on failure

---

## 🐛 Troubleshooting

### "CORS Error" on Frontend
- Make sure backend URL in Vercel env variable is correct
- Redeploy frontend after changing env var
- Check CORS is enabled in `main.py`

### "Cannot connect to backend"
- Verify Render service is running (not sleeping)
- Check backend URL is accessible: `curl <backend-url>`
- Check browser console for exact error

### "Training takes too long"
- Free Render instances are slower
- Consider upgrade or reduce `timesteps` in config
- Render free tier might auto-pause after 15 min inactivity

### "Models not saving"
- Render free tier uses ephemeral storage
- Models are lost on redeploy
- Consider S3 or persistent storage for production

---

## 💡 Optional Enhancements

### Add Training History
```python
# backend/main.py
@app.get("/training-history")
async def get_history():
    # Return past training runs
```

### Add Model Download
```python
@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    # Return trained model file
```

### Add Real-time WebSocket Updates
```python
from fastapi import WebSocket
@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    # Stream training progress in real-time
```

### Upgrade to Paid Plans
- **Vercel**: $20/month for Pro (more bandwidth, analytics)
- **Render**: $12/month for Starter (persistent storage, more resources)

---

## 📝 Important Notes

1. **Free Tier Limitations**:
   - Vercel: 100 GB bandwidth/month
   - Render: Sleeps after 15 min inactivity, 500 hours/month runtime

2. **Training on Server**:
   - Each training takes ~5 minutes
   - Free Render might timeout (increase in paid tier)
   - Consider async jobs for production

3. **Data Storage**:
   - Results saved locally on Render (ephemeral)
   - Use database for production (MongoDB, PostgreSQL)

4. **Scaling**:
   - Start with free tier to validate
   - Upgrade Render to persistent storage + GPU
   - Add Redis for caching training results

---

## ✅ Deployment Checklist (After Going Live)

- [ ] Frontend accessible at Vercel URL
- [ ] Backend healthy at Render URL
- [ ] API responds to `/` endpoint
- [ ] Can upload CSV files
- [ ] Can start training
- [ ] Results display correctly
- [ ] Improvement percentage shows
- [ ] Links work (GitHub, social)

---

## 🎉 You're Live!

Share your capstone project:
```
🚀 Live Demo: https://rl-load-balancer.vercel.app
📊 GitHub: https://github.com/YOUR_USERNAME/rl-load-balancer
📝 Paper: [Link to your capstone paper]
```

---

**Need Help?**
- Vercel Docs: https://vercel.com/docs
- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- React Docs: https://react.dev
