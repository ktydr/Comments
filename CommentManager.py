import re
from typing import Optional

import pandas as pd
import numpy as np

from Mark import Mark



class CommentManager:

    def __init__(self):
        self.tokens = dict()
        self.possibility_table = pd.DataFrame()

    def learn_from_comments(self, comments: list[str]) -> None:
        self.create_tokens(comments)
        self.create_possibility_table()

    def create_possibility_table(self) -> None:
        pass

    def create_tokens(self, comments: list[str]) -> None:
        words = list()
        for comment in comments:
            words += CommentManager.parse_comment(comment)
        self.tokenize_words(words)

    @staticmethod 
    def parse_comment(comment: str) -> (list[str], Optional[Mark]):
        words = re.findall(r"[\wäöüß]+", comment.lower())
        try:
            mark = Mark(words[-1])
            words.pop()
            return words, mark
        except:
            return words, None

    def tokenize_words(self, words: list[str]) -> None:
        pass

    def test_comments(self, comments):
        pass
