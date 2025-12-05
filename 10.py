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

episode = "10"
source = source_dir / f"[ToonsHub] {episode}.mkv"
trim = (120, None)
audio_source = source_dir / f"[Erai-raws] {episode}.mkv"

filterchain_results = filterchain(source, trim, audio_source)

if not is_preview():
    if "__main__" in dir(__main__): 
        mux(episode, filterchain_results)
    else:
        filterchain_results.final.set_output()
