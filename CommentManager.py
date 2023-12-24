import functools
import re
from typing import Optional, Any

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from Mark import Mark


class CommentManager:
    """ 
    The comment manager can be trained with training comments (single line: "comment #comment_mark"),
    predicts test comment marks (test results) for test comments (single line with #: "comment"),
    test results can be evaluated with test solutions (single line: "comment #comment_mark")

    Call the methods in the following order:
    __init__ => learn_from_comments => comments_predict => evaluate_predict
    """

    ### public interface methods

    def __init__(self, **kwargs):
        """ inits the comment manager with the config keys passed through **kwargs
        set the default config for the not set config keys
        tokenize config is turned off by default

        """
        self.tokens = None
        self.possibility_table = None
        # set default config
        self.config = {
            # the file pathes to store data (attention: doesn't create new directories!)
            "tokens_file": "stored_data/tokens.txt",
            "leader_tokens_file": "stored_data/leader_tokens.txt",
            "possibility_table_file": "stored_data/possibility_table.txt",
            "results_file": "stored_data/testResults.txt", 
            # config for the tokens function generation (turned of by default)
            "tokenize": {
                "min_common_prefix": 2, # prevents the similarity of words with opposite prefixes
                "min_dice_coef": 1  # with 1 each word becomes a token
                # tip: set min_dice_coef to 0.6 - 0.8 to obtain a good tokens function when learning
            },
            # this factor prevents unreadable small numbers when creating the possibility table
            "possibility_factor": 1000,  
            # equal factors by default
            # not equal factors heavily stimulate the predict of the certain mark
            "mark_possibility": { 
                
                Mark.NEGATIVE: 1,
                Mark.NEUTRAL: 1,
                Mark.POSITIVE: 1
            }
        }
        # overwrite config items with kwargs 
        for key, value in kwargs.items():
            self.config[key] = value

    def learn_from_comments(self, comments_path : str, relearn : bool = False) -> None:
        """  trains manager with the training comments and saves the generated function in the manager
        saves the generated functions to the file
        loads the generated functions from files if possible
        to overwrite the saved functions set the relearn flag!  

        Args:
            comments_path (str): path to the training comments file
            relearn (bool, optional): force relearn (don't use cache file and overwrite it) if the flag set True

        Returns:
            nothing
        """
        # retrieve training comments from file
        comments = CommentManager._read_comments_from_file(comments_path)
        try:  # try to load the generated functions (objects) from the stored files
            if relearn: # the relearn flag forces the learning without cache files
                raise Exception('Relearning forced!')
            # try to load tokens and possibility_table
            self.tokens = CommentManager._load_object(self.config["tokens_file"])
            self.leader_tokens = CommentManager._load_object(self.config["leader_tokens_file"])
            self.possibility_table = CommentManager._load_object(self.config["possibility_table_file"])
            print('Tokens and possibility_table loaded from the stored files')
        except Exception as e:  # if relearn is True or could not load from cache
            print(e)
            print('Start learning from training comments')
            # learn from comments logic
            # parse each comment string to comment words and comment mark
            parse_results = [self._parse_comment(comment) for comment in comments]
            # create the tokens function with all words from training comments
            self._create_tokens([words for words, _ in parse_results])
            # create the possibility table with the tokens function
            self._create_possibility_table(parse_results)
            # strore the generated functions (objects) to the files (cache)
            CommentManager._store_object(self.tokens, self.config["tokens_file"])
            CommentManager._store_object(self.leader_tokens, self.config["leader_tokens_file"])
            CommentManager._store_object(self.possibility_table, self.config["possibility_table_file"])
        # print the loaded or generated functions (objects)
        self.print_inner_state()

    def comments_predict(self, comments_path : str) -> None:
        """ predicts test comment marks using the possibility table and the tokens function
        writes results to the test results file

        Args:
            comments_path (str): path to the test comments file

        Raises:
            Exception: Error: the comment manager has not been trained yet!

        Returns:
            nothing
        """
        # a guard that check whether manager has already been trained
        if self.tokens is None and self.possibility_table is None:
            raise Exception('Error: the comment manager has not been trained yet!')
        
        # retrieve test comments from test comments file
        comments = CommentManager._read_comments_from_file(comments_path)

        # retrieve test results file from config
        with open(self.config['results_file'], 'w', encoding="utf-8") as results_file:
            print('Predictions:\n')
            for comment in comments:
                # init the assume function with 1 (neutral element for multiplication)
                assume = {}
                for mark in iter(Mark):
                    assume[mark] = 1 
                
                # retrieve the words from comment (there is no mark in test comments)
                words, _ = self._parse_comment(comment)
                for word in words:
                    if word not in self.tokens:  # if the word is unknown
                        token = self._tokenize(word)  # try to find a token for the unknown word
                        if token is None:  # if there is no suitable token then 
                            continue  # skip the unknown word (no information)
                    else: 
                        token = self.tokens[word]  # get token_id of the word
                    for mark in iter(Mark):
                        # modify the assume function with (token, mark) factor
                        assume[mark] *= self.possibility_table.loc[token, mark.value]
                        # modify the assume function with the set config mark factor  
                        assume[mark] *= self.config['mark_possibility'][mark]

                # print prediction statistics
                print(comment) 
                assume_sum = sum(list(assume.values()))
                for mark, value in assume.items():
                    print(f'{mark.value}: {round(value/assume_sum * 100, 1)} %')

                # print the best mark (comment prediction)
                best_mark = max(assume, key=assume.get) 
                print(f'=> {best_mark.value}')
                print('\n')

                # write the comment result to a file (in the parsable format)
                results_file.write(f'{comment} #{best_mark.value}\n\n')
    
    def evaluate_predict(self, solution_comments_path : str) -> None:
        """ evaluates test results using test solutions, shows evaluation statistics

        Args:
            solution_comments_path (str):  path to solution

        Raises:
            Exception: Comment mark can not be None!
            Exception: Impossible to evauluate the programm predict!

        Returns:
            nothing
        """
        # read result and solution comments from file
        result_comments = CommentManager._read_comments_from_file(self.config["results_file"])
        solution_comments = CommentManager._read_comments_from_file(solution_comments_path)

        try:
            # retrieve mark pairs to compare
            mark_compare_pairs = list(zip(
                [CommentManager._parse_comment(comment)[1] for comment in result_comments],
                [CommentManager._parse_comment(comment)[1] for comment in solution_comments]
            ))

            # init stats for each mark
            stats = {}
            for mark in iter(Mark):
                stats[mark] = {
                    'amount_in_results': 0,
                    'amount_in_solutions': 0,
                    'amount_correct': 0
                }

            # count basic stats (amount in results, amount in solution, amount correct)
            for result_mark, solution_mark in mark_compare_pairs:
                if result_mark is None or solution_mark is None:
                    raise Exception('Comment mark can not be None!')
                stats[result_mark]['amount_in_results'] += 1
                stats[solution_mark]['amount_in_solutions'] += 1
                if result_mark == solution_mark:
                    stats[result_mark]['amount_correct'] += 1  # stats[result_mark] is the same as stats[solution_mark]
            
            # calculate other stats from amounts: precision, recall and f-score
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
            
            # print stats object
            print("Prediction evaluation stats: ")
            for mark in iter(Mark):
                print(f'{mark.name} mark: ')
                for key, value in stats[mark].items():
                    print(f'\t{key}: {value}')

            # visualize and show stats
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

    def print_inner_state(self) -> None:
        """ prints tokens, leader_tokens and possibility_table of the manager 
        Returns:
            nothing
        """

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

        print('Leader tokens: ')
        for index, (word, token) in enumerate(self.leader_tokens.items()):
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
        
    def _create_possibility_table(self, parse_results : list[tuple[list[str], Optional[Mark]]]) -> None:
        """  generates the possibility table in the manager 
        initializes self.possibility_table (uses self.tokens)

        Args:
            parse_results (list[tuple[list[str], Optional[Mark]]]): list of parse results of each comment string

        Returns:
            nothing
        """

        # init mark_amount function
        mark_amount = {}
        for mark in iter(Mark):
            mark_amount[mark.value] = 0
        
        # calculate the appeared (amount) function for pairs (mark, token) 
        appeared = {}
        for words, mark in parse_results:
            for word in words:
                token = self.tokens[word]  # tokens are the attributes of the comment
                if (token, mark) not in appeared:
                    appeared[(token, mark)] = 0
                appeared[(token, mark)] += 1  # update appeared function
                mark_amount[mark.value] += 1  # update mark_amount function

        # retrieve the possibility factor from manager config
        possibility_factor = self.config["possibility_factor"]

        # create the possibility table (table data in suitable form for pd.DataFrame)
        token_values = list(set(self.tokens.values()))  # unique token_ids
        table_data = {}

        # calculate the table columns by marks
        for mark in iter(Mark):
            table_data[mark.value] = []  # init an empty column
            for token in token_values:
                if (token, mark) not in appeared:
                    appeared[(token, mark)] = 0
                # calculate the factor for (token, mark) using possibility with laplace smoothing
                p = (appeared[(token, mark)] + 0.1) / (mark_amount[mark.value] + len(token_values))
                p *= possibility_factor  # multiply the factor to avoid small numbers
                table_data[mark.value].append(p)

        # init the possibility table with pd.DataFram of table_data, tokens and marks
        self.possibility_table = pd.DataFrame(data=table_data, index=token_values)
        return

    def _create_tokens(self, comment_words: list[list[str]]) -> None:
        """ generates the tokens and the leader_tokens functions in the manager
        initializes self.tokens and self.leader_tokens

        Args:
            comment_words (list[list[str]]): all comment words listed by comment

        Returns:
            nothing
        """
        # gather all comment words in one list
        words = []
        for words_list in comment_words:
            words += words_list

        # calculate the word_entries function as a dictionary with key=unique word and value=word entry count in words
        word_entries = {}
        for word in words:
            if word not in word_entries:
                word_entries[word] = 0
            word_entries[word] += 1

        """
        - create word group for each unique word as group leader
        - each group consists of the unique words that are similar to the group leader
            (each word in the group is similar (1-similar) to its leader,
            so all words in the group are at most 2-similar to each other)
        - save the group words as a list item in word_groups_data
        - link word_group entry to word_groups_data entry with index
        """
        word_groups = []
        word_groups_data = []
        for leader_word in word_entries.keys():
            # group value is the sum of group words entry counts;
            group_value = 0  # init the sum accumulator with 0
            group_words = [] 
            for word, count  in word_entries.items(): 
                if self._are_similar(leader_word, word):
                    group_words.append(word)  # the word becomes a member of the group
                    group_value += count  # increase the group_value with the word's entry count
            # word_group entry: (group_value, group_data_index, leader_word)
            word_groups.append((group_value, len(word_groups_data), leader_word))  
            # save all the groups word in word_groups_data entry under group_data_index
            word_groups_data.append(group_words) 

        # init the manager members
        self.tokens = dict()
        self.leader_tokens = dict()

        # a comparator for groups: groups are sorted by group_value and then by the group words amount
        def compare_groups(group1, group2):
            if group1[0] > group2[0]:
                return -1
            if group1[0] < group2[0]:
                return 1
            if len(word_groups_data[group1[1]]) > len(word_groups_data[group2[1]]):
                return -1
            if len(word_groups_data[group1[1]]) < len(word_groups_data[group2[1]]):
                return 1
            return 0

        """
        - assign a token to each word
        - tokens are groups derived from word_groups, so that each words belongs exactly to one token
        - each token word is considered the same as the other token words
        - the token value in the later calculations depends on the token group_value
        - when generating tokens from word_groups prefer the word_groups entries with the largest group_value
          in order to assign words to the most powerful tokens 
        - the smaller the token id is the more valueable is the token
        """
        to_assign = len(word_entries.keys())  # words left to be assigned with a token
        token_id = 1  # the current token_id to assign
        while to_assign > 0:
            # sort the current word_groups with the comparator above
            word_groups.sort(key=functools.cmp_to_key(compare_groups)) 
            # pick the most valuable word group
            _, word_groups_data_index, leader_word = word_groups[0]
            assign_words = word_groups_data[word_groups_data_index]
            # assign the current token_id to the leader of assign_words
            self.leader_tokens[leader_word] = token_id
            # assign the current token_id to assign_words
            for word in assign_words:
                self.tokens[word] = token_id
            token_id += 1  # increase the current token_id for the next iteration
            to_assign -= len(assign_words) # subtract assign_words length from to_assign counter

            # rebuild the word_groups without assign_words
            rebuild_word_groups = []  # create a temporary word_groups object 
            for _, word_groups_data_index, leader_word in word_groups:
                # each group becomes a group without assign_words
                rebuild_group_words = [word for word in word_groups_data[word_groups_data_index]
                                        if word not in assign_words]
                word_groups_data[word_groups_data_index] = rebuild_group_words  # overwrite word_groups_data
                if rebuild_group_words:  # don't add group if there is no group members left
                    # recalculate the group_value without assign_words
                    rebuild_group_value = sum([word_entries[word] for word in rebuild_group_words])
                    # new word_group entry: (new group_value, same group_data_index, same leader_word)
                    rebuild_word_groups.append((rebuild_group_value, word_groups_data_index, leader_word)) 
            # overwrite the word_groups with the temporary word_groups 
            word_groups = rebuild_word_groups
       
        return

    def _tokenize(self, word : str) -> Optional[int]:
        """ finds the best token for the word 

        Args:
            word (str): 

        Returns:
            Optional[int]: token_id if token found, None otherwise
        """
        INF = 1e18  # define infinity constant
        # the best token must be minimized since the smaller token_id means the more token power
        best_token = INF  # best token default value is infinity

        # check all tokens: if token's leader word is similar to word then update the best_token
        for leader_word, token in self.leader_tokens.items(): 
            if (self._are_similar(word, leader_word)):
                best_token = min(best_token, token) # minimize the best_token with token

        # return the best token if best_token has been updated at least once
        if best_token != INF:
            return best_token
        return None

    def _are_similar(self, word1: str, word2: str) -> bool:
        """ checks if two words are similiar

        Args:
            word1 (str): first word
            word2 (str): second word

        Returns:
            bool: True if two words are similar, False otherwise
        """
        # special equality check for short words and words with digits
        if word1 == word2:
            return True
        
        # retrieve min common prefix and min dice coefficient from manager config
        min_common_prefix = self.config["tokenize"]["min_common_prefix"]
        min_dice_coef = self.config["tokenize"]["min_dice_coef"]

        """
        if two words don't have the minimal common prefix or 
        if one of words contains digits
        then don't compare the two words
        as they can not belong to the same word group (token)
        """
        if word1[:min_common_prefix] != word2[:min_common_prefix] or \
           bool(re.search(r"\d", word1)) or bool(re.search(r"\d", word2)):
            return False
        
        # check if the minimal dice coeficient is fulfilled 
        return CommentManager._dice(word1, word2) >= min_dice_coef
    

    ### static helper methods

    @staticmethod
    def _parse_comment(comment: str) -> tuple[list[str], Optional[Mark]]:
        """ parses a comment string to comment words and comment mark

        Args:
            comment (str): comment string

        Returns:
            tuple[list[str], Optional[Mark]]: pair containing the list of comment words and the comment mark
        """
        # use regular expressions to parse words and mark in lower_case of the comment string
        # find words as non-empty sequences of {a-z, 0-9, ä, ö, ü, ß}
        words = re.findall(r"[a-z0-9äöüß]+", comment.lower())  
        # find mark as a non-empty word beginning with # at the end of comment string
        marks = re.findall(r"#[a-z0-9äöüß]+$", comment.lower())

        if marks:  # if there is a mark then remove it from words and return the resuls
                mark = Mark(words[-1])
                words.pop()
                return words, mark
        else: # if there is no mark return the comment words with no mark
            return words, None

    @staticmethod
    def _dice(word1: str, word2: str) -> float: 
        """ calculates the dice cofficient of two words

        Args:
            word1 (str): first word
            word2 (str): second word

        Returns:
            float: the dice cofficient of two words in [0, 1]
        """
        # build trigrams of word1
        t1 = [word1[i-2: i+1] for i in range(2, len(word1))]
        # build trigrams of word2
        t2 = [word2[i-2: i+1] for i in range(2, len(word2))]
        # find common trigrams
        t_common = [item for item in t1 if item in t2]

        try: 
            # calculate the dice coefficient
            result = (2*len*t_common) / (len(t1) + len(t2))
        except:  # catch 0 division
            result = 0  # handle zero division as dice 0

        return result

    @staticmethod
    def _read_comments_from_file(path: str) -> list[str]:
        """ retrieve comment strings from file 
        each comment string is located in the single line

        Args:
            path (str): _path to file

        Returns:
            list[str]: list of comment strings
        """
        with open(path, 'r', encoding="utf-8") as file:
            comments = file.readlines() # fetch file lines as comment strings

        # remove spaces at the beginning and at the end of comment strings
        comments = [comment.strip() for comment in comments]  
        # filter comment strings as non-empty lines
        comments = [comment for comment in comments if comment]

        return comments
    
    @staticmethod 
    def _load_object(file_path : str) -> Any:
        """ Load object from binary file

        Args:
            file_path (str): path to file 

        Returns:
            Any: a python object
        """
        with open(file_path, 'rb') as object_file:
            return pickle.load(object_file)
        
    @staticmethod 
    def _store_object(obj : Any, file_path : str) -> None:
        """ Store object to file in binary mode

        Args:
            obj (Any): an object to store 
            file_path (str): relative path to file (does not create directories!)

        Returns:
            nothing
        """
        with open(file_path, 'wb') as object_file:
            return pickle.dump(obj, object_file)    
            

