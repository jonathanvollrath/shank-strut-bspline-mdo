from pathlib import Path
import yaml

# Path to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to configuration file
CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'

# Path to control points CSV file
CONTROL_POINTS_CSV = PROJECT_ROOT / 'data' / 'user' / 'control_points.csv'

# Path to holes CSV file
HOLES_CSV_PATH = PROJECT_ROOT / 'data' / 'user' / 'holes.csv'

# Path to mesh file
MESH_MSH_FILE = PROJECT_ROOT / 'data' / 'gen' / 'shank_strut.msh'
MESH_INP_FILE = PROJECT_ROOT / 'data' / 'gen' / 'shank_strut_mesh.inp'
ANALYSIS_DEF_INP_FILE = PROJECT_ROOT / 'data' / 'gen' / 'shank_strut_analysis.inp'

# Function to load default config path to external file
def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)