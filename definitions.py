from pathlib import Path
import os
from os.path import normpath, basename

ROOT_DIR_INSIDE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR_INSIDE = Path('/Users/dadua2/PROJECTS/PD_progression_subtypes')
ROOT_DIR_OUTSIDE =  Path('/Users/dadua2/PROJECTS/projects.link') / basename(ROOT_DIR_INSIDE)
