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
            words += CommentManager._parse_comment(comment)[0]
        self._tokenize_words(words)

    @staticmethod 
    def _parse_comment(comment: str) -> tuple[list[str], Optional[Mark]]:
        words = re.findall(r"[\wäöüß]+", comment.lower())
        try:
            mark = Mark(words[-1])
            words.pop()
            return words, mark
        except:
            return words, None

    def _tokenize_words(self, words: list[str]) -> None:
        self.tokens = {}

        words.sort()
        token_id = 1
        for idx, word in enumerate(words):
            for j in range(idx):
                prev_word = words[j]
                if CommentManager._are_similar(word, prev_word):
                    if word in self.tokens:
                        self.tokens[prev_word] = self.tokens[word]  # unite all similar words under one token
                    else:
                        self.tokens[word] = self.tokens[prev_word]  # inherit the token from the similar word
            if word not in self.tokens:
                self.tokens[word] = token_id
                token_id += 1
        print(self.tokens)
        return 

    @staticmethod
    def _are_similar(word: str, prev_word: str, min_common_prefix: int=2, min_dice_coef: float=0.6) -> bool:
        # use short circuit logic
        if word[:min_common_prefix] != prev_word[:min_common_prefix] or \
           bool(re.search(r"\d", word)) or bool(re.search(r"\d", prev_word)):
                return False
        if CommentManager._dice(word, prev_word) >= min_dice_coef:
            return True
        return False
    
    @staticmethod
    def _dice(word: str, prev_word: str) -> float: # modified dice coefficient
        # get trigrams of word
        t1 = {word[i-2 : i] for i in range(2, len(word))}
        # get trigrams of prev_word
        t2 = {prev_word[i-2 : i] for i in range(2, len(prev_word))}
        # calculate the dice coefficient
        try:
            # original dice_coef = 2*len(t1.intersection(t2)) / (len(t1) + len(t2))
            result = 2*len(t1.intersection(t2)) / (len(t1) + len(t2)) 
        except:
            result = 0
        return result
    
    def test_comments(self, comments):
        pass
