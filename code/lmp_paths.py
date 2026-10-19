"""Single configurable root for every data, intermediate and output path used by the scripts under code/.

LMP_DATA (environment variable) is the root. It defaults to <repository>/data, i.e. ./data when you run
from the repository root. A relative LMP_DATA is resolved against the current working directory.

Layout under $LMP_DATA (see the main README):
  l2ksu/, kapd/, s51data/, extra/, pilot/   downloaded datasets (inputs)
  full52/, full52-v2/, s54/, s55/           working folders of the acoustic, learner and early ranking stages
  results/<folder>/                         tables and JSON written by the analysis scripts
  figures/                                  figures
  tables/                                   Excel workbook written by s5.1/s51_figures_tables.py

Scripts import this module after adding code/ to sys.path relative to their own file, so they run from any
working directory. Nothing here changes analysis logic.
"""
import glob
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('LMP_DATA') or (REPO / 'data')).expanduser().resolve()
RESULTS = DATA / 'results'
FIGURES = DATA / 'figures'


def data(*parts):
    """Path (str) under $LMP_DATA; parts may contain '/'."""
    return str(DATA.joinpath(*parts))


def workdir(*parts):
    """Working folder under $LMP_DATA (created if missing), as a Path."""
    p = DATA.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(folder):
    """Output folder $LMP_DATA/results/<folder>/ (created if missing), as a str ending in a separator."""
    p = RESULTS / folder
    p.mkdir(parents=True, exist_ok=True)
    return str(p) + os.sep


def figures_dir():
    """Figure folder $LMP_DATA/figures/ (created if missing), as a str ending in a separator."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return str(FIGURES) + os.sep


def arabic_font():
    """Path of a Noto Sans Arabic font file (LMP_ARABIC_FONT, else the first match in common font folders), or None."""
    env = os.environ.get('LMP_ARABIC_FONT')
    if env and os.path.exists(env):
        return env
    dirs = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.local/share/fonts'),
            os.path.expanduser('~/.fonts'), os.path.expanduser('~/Library/Fonts'), '/Library/Fonts',
            'C:/Windows/Fonts']
    hits = sorted(f for d in dirs for f in glob.glob(os.path.join(d, '**', 'NotoSansArabic*.ttf'), recursive=True))
    pref = [f for f in hits if 'VariableFont' in f or 'Regular' in f]
    return (pref or hits or [None])[0]
