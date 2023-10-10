from CommentManager import CommentManager 

def main():

    manager = CommentManager()

    training_comments, test_comments = fetch_comments()

    manager.learn_from_comments(training_comments)

    manager.test_comments(test_comments)
    
    """
    while True:
        manager.input_comment()
    """
     
def fetch_comments():
    return ([], [])



if __name__ == "__main__":
    main()
