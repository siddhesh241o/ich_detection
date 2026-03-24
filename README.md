# VoluYOLO Triage App

Streamlit app for DICOM volume triage using a YOLO model and a 2.5D slice stack (previous, current, next).

## What this app does
- Accepts uploaded DICOM slices.
- Sorts slices by Z position.
- Builds a 2.5D image for each index using adjacent slices.
- Runs YOLO inference and shows the highest-confidence finding.

## Project structure
- `app.py` - main Streamlit app
- `requirements.txt` - Python dependencies
- `best.pt` - model weights (not committed by default)

## Setup on another PC
1. Clone the private repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Place model weights in the project root as `best.pt`, or set a custom model path with environment variable `VOLUYOLO_MODEL_PATH`.
5. Run Streamlit.

### Windows (PowerShell)
```powershell
git clone <your-private-repo-url>
cd <repo-folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Option A: put best.pt in repo root
# Option B: set custom model path
$env:VOLUYOLO_MODEL_PATH="relative path of best.pt file"
streamlit run app.py
```

### Linux / macOS
```bash
git clone <your-private-repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Option A: put best.pt in repo root
# Option B: set custom model path
export VOLUYOLO_MODEL_PATH="relative path of best.pt file"
streamlit run app.py
```

## Notes for sharing model weights
- `*.pt` is ignored in `.gitignore` to keep the repo lightweight.
- If you want the weights versioned, use Git LFS.

## Common launch issue
If you see Streamlit context warnings, make sure you launch with:
```bash
streamlit run app.py
```
(not `python app.py`).
