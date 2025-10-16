import os
import sys
sys.path.insert(0, os.getcwd())

from argparse import ArgumentParser
from vspreview import is_preview
import re
from vstools import SPath

from common import filterchain, mux

parser = ArgumentParser()
parser.add_argument("source", type=SPath, nargs="?", default=None)
args = parser.parse_args()
if args.source is not None:
    source = args.source
else:
    assert "SOURCE" in os.environ, "You need to either pass the source video via commandline parameters, or via environmental variable \"SOURCE\""
    source = SPath(os.environ["SOURCE"])

assert (m := re.search(r"e - ([\d\.]+) \(1080p\) \[", source.stem)), "Invalid source video filename"
episode = m.group(1)

filterchain_results = filterchain(source)

if not is_preview():
    mux(episode, filterchain_results)
