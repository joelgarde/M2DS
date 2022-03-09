import os
import subprocess
from pathlib import Path
from enum import Enum, auto
from tempfile import NamedTemporaryFile

NG_URL = r"http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
REUTERS_URL = r"https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz"
GLOVE_URL = r"downloads.cs.stanford.edu/nlp/data/glove.6B.zip"

class CompMode(Enum):
    TARGZ = auto()
    ZIP = auto()


def download(folder: Path, url: str, mode: CompMode = CompMode.TARGZ):
    """
    !mkdir 20NG && curl curl   | tar -xzC 20NG
    !mkdir reuters && curl  | tar -xzC reuters"
    """

    try:
        folder.mkdir(parents=True)
        if mode is CompMode.TARGZ:
            p1 = subprocess.Popen(["curl", url], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["tar", "xzC", folder], stdin=p1.stdout, stdout=subprocess.PIPE)
            p1.stdout.close()
            print(p2.communicate()[0])
    except FileExistsError:
        print("Folder already present, skipping download.")


def download_glove(glove_p: Path):
    if not glove_p.exists():
        p1 = subprocess.Popen(["curl", "-o", glove_p, GLOVE_URL])
        p2 = subprocess.Popen(["unzip", glove_p, "glove.6B.50d.txt", "-d", glove_p])

