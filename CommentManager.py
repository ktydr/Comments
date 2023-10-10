import numpy as np
import re


class CommentManager:

    def __init__(self):
        pass

    def learn_from_comments(self, comments):
        CommentManager.parse_comment("")

    @staticmethod 
    def parse_comment(comment: str) -> list[str]:
        comment = "The way 54 is too hard, but We will overwelm it!"
        words = re.findall(r"\w+", comment.lower())
        mark_name = words.pop()
        return words, mark_name

    def tokenize_words(self, words: list[str]) -> None:
        pass
