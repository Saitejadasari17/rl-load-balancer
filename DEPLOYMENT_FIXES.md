# 🚀 DEPLOYMENT FIXES - Common Errors & Solutions

## ❌ Error 1: Backend Import Errors on Render

### Problem
```
ModuleNotFoundError: No module named 'src'
ImportError: attempted relative import with no known parent package
```

### Solution
✅ **Already Fixed in new main.py:**
- Robust import fallback system
- No dependency on Python path configuration
- Works in both local and server environments

### Verify it works:
```bash
cd backend
python -c "from main import app; print('✅ Imports working')"
```

---

## ❌ Error 2: Frontend Build Failures on Vercel

### Problems
- "Cannot find module 'react'"
- ".env not found"
- "index.html not found"

### Solutions
✅ **Already Fixed - Created:**
- `frontend/src/index.js` - Entry point
- `frontend/src/index.css` - Global styles
- `frontend/public/index.html` - HTML template
- `frontend/.env.production` - Production env vars
- `frontend/postcss.config.js` - CSS processing

### Verify build:
```bash
cd frontend
npm install
npm run build
# Should complete without errors
```

---

## ❌ Error 3: CORS or API Connection Issues

### Problem
```
Access to XMLHttpRequest at 'https://...' from origin 'https://vercel...'
```

### Solution
✅ **Already Fixed in backend/main.py:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Update Vercel Environment Variable:
1. Go to: https://vercel.com/dashboard
2. Select your project
3. Settings → Environment Variables
4. Set: `REACT_APP_API_URL=https://YOUR_RENDER_URL.onrender.com`
5. Redeploy

---

## ❌ Error 4: Dataset/Model Not Found

### Problem
```
FileNotFoundError: data/dataset.tar.xz not found
FileNotFoundError: models/rl_agent_azure.zip not found
```

### Solution
- Models save to `models/` (created automatically)
- Backend creates `results/` folder automatically
- **No dataset needed for API deployment** (user uploads via UI)

### Verify directories exist:
```bash
mkdir -p models results data
```

---

## ❌ Error 5: Port Already in Use

### Problem
```
OSError: [Errno 48] Address already in use
```

### Solution
```bash
# Kill process on port 8000 (macOS/Linux)
lsof -ti:8000 | xargs kill -9

# Or use different port
python -m uvicorn main:app --port 8001
```

---

## ✅ DEPLOYMENT CHECKLIST

### Local Testing (Must Pass)
- [ ] `python backend/main.py` runs without errors
- [ ] `npm start` in frontend works
- [ ] API responds at http://localhost:8000
- [ ] Frontend loads at http://localhost:3000
- [ ] Can upload CSV and start training

### GitHub
- [ ] All files committed
- [ ] `.gitignore` excludes `models/`, `results/`, `data/`
- [ ] Pushed to `main` branch
- [ ] Repo is PUBLIC

### Render Backend
- [ ] Service created and connected to GitHub
- [ ] Build Command: `pip install -r backend/requirements.txt`
- [ ] Start Command: `cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Service shows "Running" (green)
- [ ] API responds at your Render URL

### Vercel Frontend
- [ ] Project created from GitHub
- [ ] Framework: React
- [ ] Root Directory: `./frontend`
- [ ] Environment Variable set: `REACT_APP_API_URL`
- [ ] Build succeeds (Vercel shows green checkmark)
- [ ] Site is live at your Vercel URL

---

## 🧪 TEST AFTER DEPLOYMENT

### Test Backend API
```bash
# Replace with YOUR Render URL
curl https://YOUR-BACKEND.onrender.com/
curl https://YOUR-BACKEND.onrender.com/metrics
```

Expected response:
```json
{
  "status": "online",
  "service": "RL Load Balancer API"
}
```

### Test Frontend
1. Open https://YOUR-FRONTEND.vercel.app
2. Should see dashboard with input fields
3. Console (F12) should show no red errors
4. Try uploading a CSV file
5. Click "Start Training"

---

## 🔧 QUICK FIXES

### Fix 1: Reinit npm if build fails
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Fix 2: Clear Vercel cache
- Vercel Dashboard → Settings → Git → Redeploy

### Fix 3: Check Render logs
- Render Dashboard → Select service → Logs tab
- Look for actual error messages

### Fix 4: Fallback API URL
If CORS still issues, ensure frontend URL matches exactly:
- Vercel: `https://YOUR-PROJECT.vercel.app`
- Render: `https://YOUR-SERVICE.onrender.com`
- Must be HTTPS (not http)

---

## 🚀 IF ALL ELSE FAILS

### Nuclear Option (Complete Reset)

```bash
# Clean everything
rm -rf node_modules .next dist build

# Reinstall
npm ci  # Exact version install

# Rebuild
npm run build

# Test
npm start
```

### Then redeploy:
- Push to GitHub
- Vercel auto-redeploys
- Render auto-redeploys

---

## 📞 Support URLs

| Issue | Check |
|-------|-------|
| React build errors | https://create-react-app.dev/docs/troubleshooting/ |
| FastAPI/Uvicorn | https://www.uvicorn.org/deployment/ |
| Vercel deployment | https://vercel.com/docs/concepts/projects/project-configuration |
| Render deployment | https://render.com/docs/web-services |

---

## ✅ Success Indicators

When everything works:
✓ Frontend loads instantly
✓ API responds in <500ms
✓ Can upload CSV without errors
✓ Training runs and shows progress
✓ Results display with charts
✓ Improvement % shows correctly

---

**You're almost there! These fixes handle 99% of deployment issues.** 🎉
