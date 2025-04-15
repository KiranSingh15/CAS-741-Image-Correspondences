import subprocess
from pathlib import Path

# Define source and test directories
project_root = Path(__file__).resolve().parents[1]
module_folder = project_root / "projectFiles"
test_folder = project_root / "tests"

# List all Python module files in projectFiles excluding __init__.py
modules = [p.stem for p in module_folder.glob("*.py") if p.name != "__init__.py"]

# Loop through modules and run coverage
for module in modules:
    print(f"\nRunning coverage for {module}.py ...")
    subprocess.run(
        [
            "pytest",
            f"--cov=projectFiles.{module}",
            "--cov-report=term",
            f"--cov-report=html:htmlcov/{module}",
            str(test_folder),
        ]
    )
