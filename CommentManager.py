import re
from typing import Optional, Any

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from Mark import Mark


class CommentManager:

    ### public interface methods

    def __init__(self, **kwargs):
        self.tokens = None
        self.possibility_table = None
        # set default config
        self.config = {
            "tokens_file": "stored_data/tokens.txt",
            "possibility_table_file": "stored_data/p_table.txt",
            "results_file": "stored_data/testResults.txt",
            "tokenize": {
                "min_common_prefix": 2,
                "min_dice_coef": 1  # each word becomes a token
            },
            "possibility_factor": 1000,
            "mark_possibility": { # equal factors
                Mark.NEGATIVE: 1,
                Mark.NEUTRAL: 1,
                Mark.POSITIVE: 1
            }
        }
        # overwrite config items with kwargs 
        for key, value in kwargs.items():
            self.config[key] = value

    def learn_from_comments(self, comments_path : str, relearn : bool = False) -> None:
        comments = CommentManager._read_comments_from_file(comments_path)
        try:
            if relearn:
                raise Exception('Start relearning')
            # try to load tokens and possibility_table
            self.tokens = CommentManager._load_object(self.config["tokens_file"])
            self.possibility_table = CommentManager._load_object(self.config["possibility_table_file"])
            print('Tokens and possibility_table loaded from the stored files')
        except Exception as e:
            print('Start learning from training comments')
            # learn from comments logic
            parse_results = [self._parse_comment(comment) for comment in comments]
            self._create_tokens([words for words, mark in parse_results])
            self._create_possibility_table(parse_results)
            CommentManager._store_object(self.tokens, self.config["tokens_file"])
            CommentManager._store_object(self.possibility_table, self.config["possibility_table_file"])
        self.print_inner_state()

    def comments_predict(self, comments_path : str):
        if self.tokens is None and self.possibility_table is None:
            raise Exception('Error: the comment manager has not been trained yet!')
        comments = CommentManager._read_comments_from_file(comments_path)
        with open(self.config['results_file'], 'w') as results_file:
            print('Predictions:\n')
            for comment in comments:
                assume = {}
                for mark in iter(Mark):
                    assume[mark] = 1 
                words, mark = self._parse_comment(comment)
                for word in words:
                    for mark in iter(Mark):
                        if word not in self.tokens:
                            token = self._tokenize(word)
                            if token is None:
                                continue
                        else:
                            token = self.tokens[word]
                        assume[mark] *= self.possibility_table.loc[token, mark.value] * self.config['mark_possibility'][mark]

                # generate statistics
                print(comment) 

                assume_sum = sum(list(assume.values()))
                for mark, value in assume.items():
                    print(f'{mark.value}: {round(value/assume_sum * 100, 1)} %')

                best_mark = max(assume, key=assume.get) 
         
                print(f'=> {best_mark.value}')
                print('\n')

                # write results
                results_file.write(f'{comment} #{best_mark.value}\n\n')
    
    def evaluate_predict(self, solution_comments_path):
        result_comments = CommentManager._read_comments_from_file(self.config["results_file"])
        solution_comments = CommentManager._read_comments_from_file(solution_comments_path)
        try:
            # init mark pairs to compare
            mark_compare_pairs = list(zip(
                [CommentManager._parse_comment(comment)[1] for comment in result_comments],
                [CommentManager._parse_comment(comment)[1] for comment in solution_comments]
            ))
            # init stats
            stats = {}
            for mark in iter(Mark):
                stats[mark] = {
                    'amount_in_results': 0,
                    'amount_in_solutions': 0,
                    'amount_correct': 0
                }
            # calulate basic stats
            for result_mark, solution_mark in mark_compare_pairs:
                if result_mark is None or solution_mark is None:
                    raise Exception('Comment mark can not be None!')
                stats[result_mark]['amount_in_results'] += 1
                stats[solution_mark]['amount_in_solutions'] += 1
                if result_mark == solution_mark:
                    stats[result_mark]['amount_correct'] += 1  # stats[result_mark] is the same as stats[solution_mark]
            # calculate stats: precision, recall and f-score
            for mark in iter(Mark):
                if stats[mark]['amount_in_results'] == 0:
                    stats[mark]['precision'] = 1
                else:
                    stats[mark]['precision'] = stats[mark]['amount_correct'] / stats[mark]['amount_in_results']
                if stats[mark]['amount_in_solutions'] == 0:
                    stats[mark]['recall'] = 1
                else:
                    stats[mark]['recall'] = stats[mark]['amount_correct'] / stats[mark]['amount_in_solutions']
                if stats[mark]['precision'] == 0 and stats[mark]['recall'] == 0:
                    stats[mark]['f-score'] = 0
                else:
                    stats[mark]['f-score'] = (2 * stats[mark]['precision'] * stats[mark]['recall']) / (stats[mark]['precision'] + stats[mark]['recall']) 
            # print stats
            print("Prediction evaluation stats: ")
            for mark in iter(Mark):
                print(f'{mark.name} mark: ')
                for key, value in stats[mark].items():
                    print(f'\t{key}: {value}')
            # generate and show visual statistics
            attributes = [('precision', 'Genauigkeit'),
                        ('recall', 'Treffquote'),
                        ('f-score', 'F-Maß')]
            fig, axis = plt.subplots(1, len(attributes))
            fig.tight_layout()
            fig.canvas.manager.set_window_title('Einschätzung der Vorhersage')
            for index, attr in enumerate(attributes):
                attr_key, attr_name = attr
                x_names = []
                x_values = []
                for mark in iter(Mark):
                    x_names.append(mark.value)
                    x_values.append(stats[mark][attr_key] * 100)
                x_names.append('Mittelwert')
                x_values.append(sum(x_values) / len(x_values))
                
                axis[index].bar(x_names, x_values,
                    label=[f'{round(val, 1)} %' for val in x_values],
                    color=['tab:red', 'tab:orange', 'tab:green', 'tab:blue'])
                axis[index].set_ylabel(f'{attr_name} in %')
                axis[index].set_title(attr_name)
                axis[index].legend(title='Genaue Werte')
            plt.show()
        except Exception as e:
            print("Impossible to evauluate the programm predict! Actual error:\n" + str(e))

    def print_inner_state(self):
        if self.tokens is None or self.possibility_table is None:
            print('No inner state: please train the manager!')
            return
        
        print('Comment Manager state:')

        print('Tokens for each word:')
        print('(token is a number; is considered as a comment attribute)')
        for index, (word, token) in enumerate(self.tokens.items()):
            if index >= 50:
                print('...')
                break
            print(f'\t{word}: {token}')
        print('\n')

        print('The trained possibility table:')
        print('(each entry is a factor for the attribute (token) to be with a comment mark above)')
        print(self.possibility_table.head(20))
        print('...')
        print('\n\n')

    # private inner methods
        
    def _create_possibility_table(self, parse_results) -> None:
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
        return

    def _create_tokens(self, words: list[list[str]]) -> None:

        words = []
        for words_list in words:
            words += words_list
        words.sort()

        self.tokens = dict()

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

        # retrieve token groups from self.tokens
        

        return

    def _tokenize(self, word : str) -> Optional[int]:
        return None

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

    ### static helper methods

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

    @staticmethod
    def _read_comments_from_file(path: str) -> list[str]:
        with open(path, 'r') as file:
            comments = file.readlines()
        # filter comments
        comments = [comment.strip() for comment in comments]
        comments = [comment for comment in comments if comment]
        return comments

    @staticmethod 
    def _load_object(file_path : str) -> Any:
        with open(file_path, 'rb') as object_file:
            return pickle.load(object_file)
        
    @staticmethod 
    def _store_object(obj : Any, file_path : str) -> None:
        with open(file_path, 'wb') as object_file:
            return pickle.dump(obj, object_file)    
            

