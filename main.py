from CommentManager import CommentManager 

def main():

    manager = CommentManager()

    training_comments, test_comments = fetch_comments()

    manager.learn_from_comments(training_comments)

    manager.comments_predict(test_comments)
    
    """
    while True:
        manager.input_comment()
    """
     
def fetch_comments():
    # fetch training comments
    training_file = open('trainingComments.txt', 'r')
    training_comments = training_file.readlines()
    training_comments = [comment for comment in training_comments if comment != '\n']
    training_file.close()

    # fetch test comments
    test_file = open('testComments.txt', 'r')
    test_comments = test_file.readlines()
    test_comments = [comment for comment in test_comments if comment != '\n']
    test_file.close()

    return (training_comments, test_comments)



if __name__ == "__main__":
    main()
