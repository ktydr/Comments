import re
from typing import Optional

import pandas as pd
import numpy as np

from Mark import Mark


class CommentManager:

    def __init__(self):
        self.tokens = dict()
        self.possibility_table = pd.DataFrame()
        self.config = {
            "tokenize": {
                "min_common_prefix": 2,
                "min_dice_coef": 0.8
            },
            "possibility_factor": 1000,
            "mark_possibility": {
                Mark.NEGATIVE: 0.3,
                Mark.NEUTRAL: 0.4,
                Mark.POSITIVE: 0.3
            }
        }

    def learn_from_comments(self, comments: list[str]) -> None:
        parse_results = [self._parse_comment(comment) for comment in comments]
        self.create_tokens([words for words, mark in parse_results])
        self.create_possibility_table(parse_results)

    def create_possibility_table(self, parse_results) -> None:
        # init mark_amount dictionary
        mark_amount = {}
        for mark in iter(Mark):
            mark_amount[mark.value] = 0
        
        appeared = {}
        for words, mark in parse_results:
            for word in words:
                token = self.tokens[word]
                if (token, mark) not in appeared:
                    appeared[(token, mark)] = 0
                appeared[(token, mark)] += 1
                mark_amount[mark.value] += 1

        # create the possibility table
        possibility_factor = self.config["possibility_factor"]
        token_values = list(set(self.tokens.values()))
        table_data = {}

        for mark in iter(Mark):
            table_data[mark.value] = []
            for token in token_values:
                if (token, mark) not in appeared:
                    appeared[(token, mark)] = 0
                # laplace smoothing
                p = (appeared[(token, mark)] + 0.1) / (mark_amount[mark.value] + len(token_values))
                p *= possibility_factor
                table_data[mark.value].append(p)

        self.possibility_table = pd.DataFrame(data=table_data, index=token_values)
        print(self.possibility_table)

    def create_tokens(self, words: list[list[str]]) -> None:
        all_words = []
        for words_list in words:
            all_words += words_list
        self._tokenize_words(all_words)

    @staticmethod
    def _parse_comment(comment: str) -> tuple[list[str], Optional[Mark]]:
        words = re.findall(r"[a-z0-9äöüß]+", comment.lower())
        marks = re.findall(r"#[a-z0-9äöüß]+$", comment.lower())
        if marks:
                mark = Mark(words[-1])
                words.pop()
                return words, mark
        else:
            return words, None

    def _tokenize_words(self, words: list[str]) -> None:
        self.tokens = {}

        words.sort()
        token_id = 1
        for idx, word in enumerate(words):
            for j in range(idx):
                prev_word = words[j]
                if self._are_similar(word, prev_word):
                    if word in self.tokens:
                        # unite all similar words under one token
                        self.tokens[prev_word] = self.tokens[word]
                    else:
                        # inherit the token from the similar word
                        self.tokens[word] = self.tokens[prev_word]
            if word not in self.tokens:
                self.tokens[word] = token_id
                token_id += 1
        print(self.tokens)
        return

    def _are_similar(self, word: str, prev_word: str) -> bool:
        min_common_prefix = self.config["tokenize"]["min_common_prefix"]
        min_dice_coef = self.config["tokenize"]["min_dice_coef"]
        # use short circuit logic
        if word[:min_common_prefix] != prev_word[:min_common_prefix] or \
           bool(re.search(r"\d", word)) or bool(re.search(r"\d", prev_word)):
            return False
        if CommentManager._dice(word, prev_word) >= min_dice_coef:
            return True
        return False

    @staticmethod
    def _dice(word: str, prev_word: str) -> float:  # modified dice coefficient
        # get trigrams of word
        t1 = {word[i-2: i] for i in range(2, len(word))}
        # get trigrams of prev_word
        t2 = {prev_word[i-2: i] for i in range(2, len(prev_word))}
        # calculate the dice coefficient
        try:
            # original dice_coef = 2*len(t1.intersection(t2)) / (len(t1) + len(t2))
            result = 2*len(t1.intersection(t2)) / (len(t1) + len(t2))
        except:
            result = 0
        return result

    def comments_predict(self, comments):
        for comment in comments:
            assume = {}
            for mark in iter(Mark):
                assume[mark] = self.config['mark_possibility'][mark]
            words, mark = self._parse_comment(comment)
            for word in words:
                for mark in iter(Mark):
                    if word not in self.tokens:
                        continue
                    assume[mark] *= self.possibility_table.loc[self.tokens[word], mark.value]

            # generate statistics
            assume_sum = sum(list(assume.values()))
            for mark, value in assume.items():
                print(f'{mark.value}: {round(value/assume_sum * 100, 1)} %')

            best_mark = max(assume, key=assume.get)           
            print(best_mark.value)
            print('\n')
