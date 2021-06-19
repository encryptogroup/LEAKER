# Imports random library
import random

# Import pandas and numpy libraries which are used to express matrices
import pandas as pd
import numpy as np

# Class MatrixGenerator gives a user the tools to:
# 1) Generate an inverted index (words on the y-axis, documents on the x-axis) from a list of documents and
#   a dictionary of words and lists of the documents each words occurs in at least once
# 2) Generate a cooccurrence matrix from a given inverted index
# 3) Add Gaussian noise to a given cooccurrence matrix
class MatrixGenerator:

    # Method generate_inverted_index generates an inverted index from a list of documents and a dictionary of words and
    #   the lists of documents each word occurs in
    def generate_inverted_index(self, files_per_word_occurrence, document_list):

        # Randomizes the list of documents
        documents = random.sample(document_list, len(document_list))

        # Randomizes the list of words
        words = list(files_per_word_occurrence.keys())
        index = random.sample(words, len(words))

        # Creates a pandas DataFrame with the words on the y-axis and documents on the x-axis and False in every cell
        df = pd.DataFrame(
            np.full((len(index), len(documents)), False), index=index, columns=documents
        )

        # Loops through all words in the word occurrence dictionary
        for word, files in files_per_word_occurrence.items():

            # Loops through all files belonging to a single word and sets the corresponding cell value to True
            for file in files:
                # Worst mistake of my life ever, apparently .loc takes way more time than df[file][word]
                # df.loc[word, file] = True
                df[file][word] = True
        return df.copy()

    # Method generate_cooccurrence_matrix generates a cooccurrence matrix from a given inverted index, adds Gaussian noise if required
    def generate_cooccurrence_matrix(
        self, inverted_index, gaussian_noise_constant=None
    ):
        # Number of documents and words respectively
        num_docs = len(inverted_index.columns.tolist())
        num_words = len(inverted_index.index.tolist())

        # Loops through an inverted index, per word returns a list containing ones and zeros
        #   (corresponding to True/False in the inverted index respectively)
        # Zeros and ones are easier to perform calculations on
        index = []
        for i, (word, row) in enumerate(inverted_index.iterrows()):
            index.append([1 if x == True else 0 for x in list(row)])

        result_array = []

        # Loops through all words
        for i in range(0, num_words):
            result_word = []
            row_1 = index[i]

            # If the word is not the first word (meaning that result_array is empty) the algorithm already takes already computed values from
            #   the result_array (as the cooccurrence matrix is symmetric)
            # For example, when calculating the cooccurrence counts for word_1, we first calculate the cooccurrence count of ẁord_1 and word_1 and
            #   then of word_1 and word_2 and so on
            # When calculating the cooccurrence counts of word_2 we start with the cooccurrence count of word_2 and word_1,
            #    which we can re-use as we calculated it earlier
            if i != 0:
                result_word += [x[i] for x in result_array]

            # The cooccurrence count of a word with itself is added to the result_word list
            result_word.append(sum(row_1))

            # The cooccurrence count of a word with all words after its index in the words list is calculated and added to the result_word list
            for j in range(i + 1, num_words):
                row_2 = index[j]
                count = sum(
                    [
                        1
                        for k in range(0, len(row_1))
                        if row_1[k] == row_2[k] and row_1[k] == 1
                    ]
                )
                result_word.append(count)
            result_array.append(result_word)

        # The entire generated matrix is divided by the total number of documents to get a relative cooccurrence count and not an actual cooccurrence count
        result_array = np.array(result_array) / num_docs

        # Adds Gaussian noise is requested
        if gaussian_noise_constant is not None:
            result_array = self.add_gaussian_noise(
                result_array, gaussian_noise_constant
            )

        # Adds the numpy array to a pandas dataframe and sets the indices for easy look up during the IKK run
        result_dataframe = pd.DataFrame(
            result_array,
            index=list(inverted_index.index),
            columns=list(inverted_index.index),
        )
        return result_dataframe.copy()

    # Adds Gaussian noise to a given (background) cooccurrence matrix
    # It is possible that, due to the addition of noise that a cooccurrence value becomes zero, this is not prohibited at least
    # This does not reflect a real-life scenario though
    def add_gaussian_noise(
        self, background_knowledge_cooccurrence_matrix, gaussian_noise_constant
    ):

        # A value set by the authors of the IKK paper ranging between 0 and 1.0
        # 0 corresponds with no noise added to the matrix
        C = gaussian_noise_constant

        # Finds the unique values in the cooccurrence matrix, unique meaning it removes the values that are present twice due to the similarity of the matrix
        set_values = []
        for i in range(0, len(background_knowledge_cooccurrence_matrix)):
            for j in range(0, len(background_knowledge_cooccurrence_matrix[0])):
                if j >= i:
                    set_values.append(background_knowledge_cooccurrence_matrix[i][j])

        # mu is the average of all 'unique' values in the cooccurrence matrix
        mu = sum(set_values) / len(set_values)

        # difference_values is mu subtracted from every single value
        difference_values = [x - mu for x in set_values]

        # sigma_squared squares all values in difference_values and summates them before dividing by the total number of 'unique' values to get an average
        sigma_squared = sum([x ** 2 for x in difference_values]) / len(set_values)

        # noise is a list of the same amount of values as the 'unique' values by randomly picking values from a certain Normal distribution with
        #   a mu of 0 and a variance of C * sigma_squared
        noise = np.random.normal(0, C * sigma_squared, len(set_values))

        # new_values is the list of 'unique' values set_values with the addition of the noise
        new_values = [set_values[i] + noise[i] for i in range(0, len(set_values))]

        # noisy_matrix is the upper half (diagonally) of the cooccurrence matrix with noise addition
        # All values which we did not fill yet (lower half diagonally) in the matrix are set to 0
        noisy_matrix = []
        current_count = len(background_knowledge_cooccurrence_matrix)
        current_pointer = 0
        for i in range(0, len(background_knowledge_cooccurrence_matrix)):
            row = new_values[current_pointer : current_pointer + current_count]
            row = [
                0
                for x in range(
                    0, len(background_knowledge_cooccurrence_matrix) - len(row)
                )
            ] + row
            noisy_matrix.append(row)
            current_pointer += current_count
            current_count -= 1

        # noise_matrix_reversed is a transposition of the noisy_matrix
        # This matrix is used easy generate the full cooccurrence matrix
        noisy_matrix_reversed = []
        for i in range(0, len(noisy_matrix)):
            noisy_matrix_reversed.append([x[i] for x in noisy_matrix])

        result_matrix = []
        # Aggregrates noisy_matrix and noisy_matrix_reversed to generate the full result_matrix
        for row in range(0, len(noisy_matrix)):
            result_row = []
            for column in range(0, len(noisy_matrix)):
                # row == column denotes a diagonal and thus this is a special case
                if row == column:
                    result_row.append(noisy_matrix[row][column])
                else:
                    result_row.append(
                        noisy_matrix[row][column] + noisy_matrix_reversed[row][column]
                    )
            result_matrix.append(result_row)
        background_knowledge_cooccurrence_matrix = np.array(result_matrix)
        return background_knowledge_cooccurrence_matrix
