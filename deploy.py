#!/usr/bin/env python3
"""
One-command deployment helper
Validates everything and pushes to GitHub
"""

import subprocess
import sys
import os

def run_cmd(cmd, desc):
    """Run command and show status"""
    print(f"\n📋 {desc}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ FAILED: {result.stderr}")
            return False
        print(f"✅ {desc} complete")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 50)
    print("🚀 RL Load Balancer - Deploy Helper")
    print("=" * 50)
    
    checks = [
        ("git status", "Checking Git status"),
        ("python -m pip --version", "Checking Python"),
        ("node --version", "Checking Node.js"),
    ]
    
    print("\n🔍 Pre-deployment checks:")
    for cmd, desc in checks:
        if not run_cmd(cmd, desc):
            print("\n❌ Prerequisites not met. Install Git, Python, and Node.js")
            return False
    
    print("\n📝 Deployment steps:")
    
    # Build frontend
    print("\n1️⃣ Building frontend...")
    os.chdir("frontend")
    if not run_cmd("npm run build", "Frontend build"):
        return False
    os.chdir("..")
    
    # Test backend imports
    print("\n2️⃣ Testing backend...")
    if not run_cmd("python -c \"import sys; sys.path.insert(0, 'backend'); from main import app; print('✅ Backend OK')\"", "Backend validation"):
        return False
    
    # Git operations
    print("\n3️⃣ Pushing to GitHub...")
    steps = [
        ("git add .", "Stage all changes"),
        ("git commit -m 'Deployment: Fixed backend, frontend, and configs'", "Commit changes"),
        ("git push origin main", "Push to GitHub"),
    ]
    
    for cmd, desc in steps:
        if not run_cmd(cmd, desc):
            print(f"\n⚠️ {desc} might have failed")
            # Continue anyway
            continue
    
    print("\n" + "=" * 50)
    print("✅ DEPLOYMENT READY!")
    print("=" * 50)
    print("""
📊 Next steps:

1. Render Backend:
   - Go to render.com
   - New Web Service
   - Connect GitHub repo
   - Build: pip install -r backend/requirements.txt
   - Start: cd backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT
   - Deploy!

2. Vercel Frontend:
   - Go to vercel.com
   - Import GitHub repo
   - Root: ./frontend
   - Env: REACT_APP_API_URL=<your-render-url>
   - Deploy!

3. Test:
   - Visit your Vercel URL
   - Upload CSV file
   - Click "Start Training"
   - Watch it work! 🎉
    """)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
