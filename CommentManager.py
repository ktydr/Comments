import pandas as pd
from typing import Optional
import re

from Mark import Mark


class CommentManager:

    def __init__(self):
        self.tokens = dict()
        self.possibility_table = pd.DataFrame()

    def learn_from_comments(self, comments: list[str]) -> None:
        self.create_tokens(comments)
        self.create_possibility_table(comments)

    def create_possibility_table(self, comments) -> None:
        appeared = {}
        for comment in comments:
            words, mark = self.parse_comment(comment)
            for word in words:
                for mark in iter(Mark):
                    token = self.tokens[word]
                    if (token, mark) not in appeared:
                        appeared[(token, mark)] = 0
                    appeared[(token, mark)] += 1
        
        data = []    
        token_values = list(set(self.tokens.values()))
        for token in token_values:
            # summarized = prob0 = prob1 = prob2 = 0
            sum = 0
            for mark in iter(Mark):
                if (token, mark) not in appeared:
                    appeared[(token, mark)] = 0
                sum += appeared[(token, mark)]
            
            row = {}
            for mark in iter(Mark):
                try:
                    row[mark.value] = appeared[(token, mark)] / sum
                except:
                    row[mark.value] = 0
            
            data.append(row)
            
        self.create_possibility_table = pd.DataFrame(data=data, index=token_values)


    def create_tokens(self, comments: list[str]) -> None:
        words = list()
        for comment in comments:
            words += CommentManager.parse_comment(comment)
        self.tokenize_words(words)
        
    @staticmethod 
    def parse_comment(comment: str) -> tuple[list[str], Optional[Mark]]:
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
