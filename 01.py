import os
import sys
sys.path.insert(0, os.getcwd())

import __main__

from argparse import ArgumentParser
from vspreview import is_preview
from vstools import SPath

from common import filterchain, mux

parser = ArgumentParser()
parser.add_argument("--source-dir", type=SPath, default=None)
args = parser.parse_args()
if args.source_dir is not None:
    source_dir = args.source_dir
else:
    assert "SOURCE_DIR" in os.environ, "You need to either pass the directory containing the sources via commandline parameters `--source-dir`, or via environmental variable \"SOURCE_DIR\""
    source_dir = SPath(os.environ["SOURCE_DIR"])

episode = "01"
source = source_dir / f"[VARYG] {episode}.mkv"
trim = (132, None)

filterchain_results = filterchain(source, trim)

if not is_preview():
    if "__main__" in dir(__main__): 
        mux(episode, filterchain_results)
    else:
        filterchain_results.final.set_output()
