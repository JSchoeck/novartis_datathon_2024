import sys
from pathlib import Path

# Add the path of the parent directory to the system path, so that scripts from ./src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("src")))
