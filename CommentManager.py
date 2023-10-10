import pandas as pd


from Mark import Mark
import numpy as np
import re


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
    def parse_comment(comment: str) -> list[str]:
        words = re.findall(r"[\wäöüß]+", comment.lower())
        mark_name = words.pop()
        return words, mark_name

    def tokenize_words(self, words: list[str]) -> None:
        pass
