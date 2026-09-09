"""Location settings for the explicitly invoked shared experiment modules."""
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("SHARED_PROJECT_ROOT", PROJECT_ROOT))).resolve()
VJEPA_ROOT = Path(os.environ.get("VJEPA_ROOT", PROJECT_ROOT / "vjepa2")).resolve()
