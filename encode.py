from argparse import ArgumentParser
from vspreview import is_preview
from vstools import SPath

from common import filterchain, mux

parser = ArgumentParser()
parser.add_argument("source", type=SPath)
args = parser.parse_args()

filterchain_results = filterchain(source=args.source)

if not is_preview():
    mux(filterchain_results=filterchain_results)
