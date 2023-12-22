from CommentManager import CommentManager 

def main():

    manager = CommentManager()

    manager.learn_from_comments(
        read_comments_from_file('trainingComments.txt')
    )

    manager.comments_predict(
        read_comments_from_file('testComments.txt')
    )

    CommentManager.evaluate_predict(
        read_comments_from_file('testResults.txt'),
        read_comments_from_file('testSolutions.txt')
    )

    
    
    """
    while True:
        manager.input_comment()
    """

def read_comments_from_file(path: str) -> list[str]:
    file = open(path, 'r') 
    comments = file.readlines()
    file.close()
    # filter comments
    comments = [comment.strip() for comment in comments]
    comments = [comment for comment in comments if comment]
    return comments



if __name__ == "__main__":
    main()
