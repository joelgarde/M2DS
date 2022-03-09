from abc import ABC
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple

from src.consts import REUTERS_P


class ReutersParser(HTMLParser):
    """get the text and topics from the reuters files."""

    def error(self, message):
        print(message)

    def __init__(self):
        super().__init__()
        self.splits = list()
        self.text = list()
        self.topics = list()
        self.current_topics = list()
        self.current_text = ""
        self.in_topics = False
        self.in_text = False
        self.n_docs = 0

    def handle_starttag(self, tag, attrs):
        self.n_docs += (tag == "reuters")
        self.in_topics |= (tag == "topics")
        self.in_text |= (tag == "text")
        if tag == "reuters":
            self.splits.append(dict(attrs)["lewissplit"])

    def handle_endtag(self, tag):
        if tag == "text":
            self.in_text = False
            self.text.append(self.current_text)
            self.current_text = ""

        elif tag == "topics":
            self.in_topics = False
            self.topics.append(self.current_topics)
            self.current_topics = list()

    def handle_data(self, data):
        if self.in_topics:
            self.current_topics.append(data)
        elif self.in_text:
            self.current_text += data

    def __iter__(self):
        yield from zip(self.text, self.topics, self.splits)


@dataclass
class ReutersDataset:
    text: tuple
    topics: tuple
    splits: tuple

    def __getitem__(self, i):
        return self.text[i], self.topics[i], self.splits[i]

    def __len__(self):
        return len(self.text)


def reuters() -> ReutersDataset:
    full_text = ""
    for doc in REUTERS_P.iterdir():
        if doc.suffix == ".sgm":
            with open(doc, encoding="8859") as f:
                full_text += f.read()
    parser = ReutersParser()
    parser.feed(full_text)
    return ReutersDataset(tuple(parser.text), tuple(parser.topics), tuple(parser.splits))