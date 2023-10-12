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
    training_comments = read_comments_from_file('trainingComments.txt')
    test_comments = read_comments_from_file('testComments.txt')
    return (training_comments, test_comments)

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
