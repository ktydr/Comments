from CommentManager import CommentManager 

from Mark import Mark


def main():

    manager = CommentManager(tokenize={
                "min_common_prefix": 2,
                "min_dice_coef": 0.7
            })

    manager.learn_from_comments('input_data/trainingComments.txt')

    manager.comments_predict('input_data/testComments.txt')

    manager.evaluate_predict('input_data/testSolutions.txt')


if __name__ == "__main__":
    main()
