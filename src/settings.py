from pathlib import Path
import yaml

# Path to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to configuration file
CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'

# Path to control points CSV file
CONTROL_POINTS_CSV = PROJECT_ROOT / 'data' / 'control_points.csv'

# Function to load default config path to external file
def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)