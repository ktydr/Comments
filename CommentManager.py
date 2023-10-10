import numpy as np
import re


class CommentManager:

    def __init__(self):
        pass

    def learn_from_comments(self, comments):
        CommentManager.parse_comment("")

    @staticmethod 
    def parse_comment(comment: str) -> list[str]:
        words = re.findall(r"[\wäöüß]+", comment.lower())
        mark_name = words.pop()
        return words, mark_name

    def tokenize_words(self, words: list[str]) -> None:
        pass
