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

    def tokenize_words(self, words: list[str], min_common_prefix: int=2, min_dice_coef: float=0.5) -> None:
        self.tokens = {}

        words.sort()
        token_id = 1
        for idx, word in enumerate(words):
            for j in range(idx):
                prev_word = words[j]
                if word[:min_common_prefix] == prev_word[:min_common_prefix] and \
                   (word.startswith(prev_word) or CommentManager.dice(word, prev_word) >= min_dice_coef):
                        self.tokens[word] = self.tokens[prev_word]
            if word not in self.tokens:
                self.tokens[word] = token_id
                token_id += 1
        return     

    def test_comments(self, comments):
        pass
