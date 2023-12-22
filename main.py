from CommentManager import CommentManager 

def main():

    manager = CommentManager()

    manager.learn_from_comments('input_data/trainingComments.txt')

    manager.comments_predict('input_data/testComments.txt')

    manager.evaluate_predict('input_data/testSolutions.txt')

if __name__ == "__main__":
    main()
