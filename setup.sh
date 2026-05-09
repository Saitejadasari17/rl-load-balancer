#!/bin/bash
set -e

echo "🚀 RL Load Balancer Deployment Setup"
echo "===================================="
echo ""

# Check requirements
echo "✓ Checking Git..."
git --version > /dev/null

echo "✓ Checking Node.js..."
node --version > /dev/null
npm --version > /dev/null

echo "✓ Checking Python..."
python --version > /dev/null

echo ""
echo "✅ All prerequisites installed!"
echo ""

# Create environment files
echo "📝 Creating environment files..."

cat > frontend/.env.local << EOF
REACT_APP_API_URL=http://localhost:8000
EOF

cat > backend/.env << EOF
PYTHONUNBUFFERED=1
EOF

echo "✅ Environment files created!"
echo ""

echo "📋 Next steps:"
echo "1. Backend:  cd backend && python -m uvicorn main:app --reload"
echo "2. Frontend: cd frontend && npm install && npm start"
echo "3. Visit: http://localhost:3000"
echo ""
echo "🚀 Ready to develop!"
