import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def vectorize_stories(data, word_index, max_story_len, max_question_len):
    '''
    INPUT:

    data: consisting of Stories,Queries,and Answers
    word_index: word index dictionary from tokenizer
    max_story_len: the length of the longest story (used for pad_sequences function)
    max_question_len: length of the longest question (used for pad_sequences function)


    OUTPUT:

    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.

    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    '''


    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []


    for story, query, answer in data:

        # Grab the word index for every word in story
        x = [word_index[word.lower()] for word in story]
        # Grab the word index for every word in query
        xq = [word_index[word.lower()] for word in query]

        # Grab the Answers (either Yes/No so we don't need to use list comprehension here)
        # Index 0 is reserved so we're going to use + 1
        y = np.zeros(len(word_index) + 1)

        # Now that y is all zeros and we know its just Yes/No , we can use numpy logic to create this assignment
        y[word_index[answer]] = 1

        # Append each set of story,query, and answer to their respective holding lists
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # Finally, pad the sequences based on their max length so the RNN can be trained on uniformly long sequences.

    # RETURN TUPLE FOR UNPACKING
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


def create_vectorization_from_input(story, question, correct_answer, tokenizer_data):
    # Set data to expected model input shape
    my_data = [(story.split(), question.split(), correct_answer)] # Story, question and answer

    # Vectorize text
    return vectorize_stories(my_data,
                             tokenizer_data["word_index"],
                             tokenizer_data["max_story_len"],
                             tokenizer_data["max_question_len"]
                             )

def parse_result_from_model(results, tokenizer_word_index):
    # Get prediction with max certainty
    result = np.argmax(results[0])

    # Get the actual parsed tokenizer value
    for key, val in tokenizer_word_index.items():
        if val == result:
            result_text = key

    return result_text, results[0][result]