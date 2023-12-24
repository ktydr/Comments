from CommentManager import CommentManager 
from Mark import Mark


def main():
    # create comment manager (pass config keys as kwargs)
    manager = CommentManager(tokenize={
                "min_common_prefix": 2,
                "min_dice_coef": 0.7
            })

    # train manager on the training comments
    manager.learn_from_comments('input_data/trainingComments.txt', relearn=False)

    # predict test comments marks using the trained manager
    manager.comments_predict('input_data/testComments.txt')

    # evaluate the predict results with test solutions
    manager.evaluate_predict('input_data/testSolutions.txt')


if __name__ == "__main__":
    main()
