#!/bin/bash
# Chloe AI Setup Script

echo "🤖 Chloe AI - Automated Setup"
echo "=============================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs workspace
mkdir -p data/chroma data/chroma_enhanced

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Ollama setup required - see instructions below!"
else
    echo "✅ .env file already exists"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Run system test
echo "🧪 Running system test..."
python comprehensive_test.py

if [ $? -eq 0 ]; then
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "🚀 Next steps:"
    echo "1. Install Ollama: https://ollama.ai"
    echo "2. Run Ollama server: ollama serve"
    echo "3. Pull a model: ollama pull llama2"
    echo "4. Run 'python cli.py' for interactive mode"
    echo "5. Run 'python cli.py --mode api' for API server"
    echo "6. Check test_results.json for detailed system status"
else
    echo "⚠️  Setup completed with some issues"
    echo "Check the test output above for details"
fi