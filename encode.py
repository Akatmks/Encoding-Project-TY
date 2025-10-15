from argparse import ArgumentParser
import os
from vspreview import is_preview
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

filterchain_results = filterchain(source=source)

if not is_preview():
    mux(filterchain_results=filterchain_results)
