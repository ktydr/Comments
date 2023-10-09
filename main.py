from CommentManager import CommentManager 

def main():

    manager = CommentManager()

    comments = fetch_training_comments()
    test_comments = fetch_test_comments()

    manager.learn_from_comments(comments)

    manager.test_comments(test_comments)
    
    """
    while True:
        manager.input_comment()
    """
     
def fetch_training_comments():
    return []

def fetch_test_comments ():
    return []


if __name__ == "__main__":
    main()
