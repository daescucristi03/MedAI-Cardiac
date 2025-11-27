if ! command -v brew &> /dev/null
then
echo “Homebrew nu este instalat. Instalează-l cu:”
echo ‘/bin/bash -c “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)”’
exit
fi

echo “Installing Python via Homebrew…”
brew install python

echo “Creating virtual environment…”
python3 -m venv venv

echo “Activating venv…”
source venv/bin/activate

echo “📦 Installing Python libraries…”
pip install –upgrade pip
pip install numpy pandas scikit-learn joblib fastapi uvicorn python-multipart jupyterlab

echo “✨ All dependencies installed successfully!”
echo “To activate environment later, run:”
echo “    source venv/bin/activate”
