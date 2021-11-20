import abc
import itertools
import operator
import random
import string
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from datautils.models import TrainingParameters


class LinearClassifier(abc.ABC):

    def __init__(self, n_features: int):
        self.character_templates: Dict[str, np.ndarray] = {letter: np.random.uniform(0, 1, (n_features,)) for letter in
                                                           string.ascii_lowercase}
        self.biases: Dict[str, int] = {letter: random.randint(0, 1) for letter in string.ascii_lowercase}

    def train(self, X: List[np.ndarray], y: List[str], parameters: TrainingParameters = TrainingParameters()):
        continue_training = True
        epoch_number = 1
        mistake_holder: List[int] = []
        while continue_training:
            continue_training, n_mistakes = self.train_single_epoch(X, y, epoch_number, parameters.LearningRate)
            mistake_holder.append(n_mistakes)
            epoch_number += 1

    @abc.abstractmethod
    def train_single_epoch(self, X: List[np.ndarray], y: List[str], epoch_number: int, lr: float):
        return True, 0

    @abc.abstractmethod
    def test(self, X_test: List[np.ndarray], y_test: List[str]):
        pass

    @abc.abstractmethod
    def classify(self, example: np.ndarray):
        pass

    def calculate_sequence_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        err = 0
        for example, true_class in zip(X_test, y_test):
            prediction = self.classify(example)
            if prediction != true_class:
                err += 1
        print("Total Sequence error is {} incorrect out of {} sequences: {} %".format(err, len(y_test),
                                                                                      err / len(y_test) * 100))

    def calculate_char_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        err = 0
        characters = 0
        for example, true_class in zip(X_test, y_test):
            prediction = self.classify(example)
            characters += len(true_class)
            for l1, l2 in zip(true_class, prediction):
                if l1 != l2:
                    err += 1
        print("Total Character error is {} incorrect out of {} characters: {} %".format(err, characters,
                                                                                        err / characters * 100))


class BasicLinearClassifier(LinearClassifier):

    def __init__(self, n_features: int):
        super().__init__(n_features)

    def train(self, X: List[np.ndarray], y: List[str], parameters: TrainingParameters = TrainingParameters()):
        print("Training {}".format(self.__class__.__name__))
        return super(BasicLinearClassifier, self).train(X, y, parameters)

    def train_single_epoch(self, X: List[np.ndarray], y: List[str], epoch_number: int, lr: float) -> Tuple[bool, int]:
        mistakes_in_epoch: int = 0
        for i, (example, true_class) in enumerate(zip(X, y)):
            for letter_index in range(len(true_class)):
                final_prediction: str = self.pick_letter_with_highest_score(example[:, letter_index])
                if not final_prediction == true_class[letter_index]:
                    self.character_templates[true_class[letter_index]] += lr * example[:, letter_index]
                    self.character_templates[final_prediction] -= lr * example[:, letter_index]
                    mistakes_in_epoch += 1
        print("There was {} mistakes in epoch {}".format(mistakes_in_epoch, epoch_number))
        return mistakes_in_epoch != 0, mistakes_in_epoch

    def pick_letter_with_highest_score(self, letter_to_predict: np.ndarray):
        max_score = float('-inf')
        final_prediction: Optional[str] = None
        single_letter_features = letter_to_predict
        for possible_prediction in string.ascii_lowercase:
            score: float = np.dot(single_letter_features, self.character_templates[possible_prediction]) + \
                           self.biases[possible_prediction]
            if score > max_score:
                max_score = score
                final_prediction = possible_prediction
        return final_prediction

    def classify(self, example: np.ndarray):
        final_prediction = ""
        for letter_index in range(example.shape[1]):
            prediction: str = self.pick_letter_with_highest_score(example[:, letter_index])
            final_prediction += prediction
        return final_prediction

    def test(self, X_test: List[np.ndarray], y_test: List[str]):
        self.calculate_sequence_prediction_error(X_test, y_test)
        self.calculate_char_prediction_error(X_test, y_test)

    def calculate_sequence_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        return super(BasicLinearClassifier, self).calculate_sequence_prediction_error(X_test, y_test)

    def calculate_char_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        return super(BasicLinearClassifier, self).calculate_char_prediction_error(X_test, y_test)


class FixedNSequencesLinearClassifier(LinearClassifier):

    def __init__(self, n_features: int):
        super().__init__(n_features)
        self.apriori_probability = defaultdict(lambda: 0)

    def train(self, X: List[np.ndarray], y: List[str], parameters: TrainingParameters = TrainingParameters()):
        print("Training {}".format(self.__class__.__name__))
        self.memorize_prior_distribution(y)
        return super(FixedNSequencesLinearClassifier, self).train(X, y, parameters)

    def train_single_epoch(self, X: List[np.ndarray], y: List[str], epoch_number: int, lr: float) -> Tuple[bool, int]:
        mistakes_in_epoch: int = 0
        for i, (example, true_class) in enumerate(zip(X, y)):
            final_prediction = self.classify(example)
            if not final_prediction == true_class:
                for j, (predicted, real) in enumerate(zip(final_prediction, true_class)):
                    self.character_templates[real] += lr * example[:, j]
                    self.character_templates[predicted] -= lr * example[:, j]
                    self.biases[real] += lr
                    self.biases[predicted] -= lr

                mistakes_in_epoch += 1
        print("There was {} mistakes in epoch {}".format(mistakes_in_epoch, epoch_number))
        return mistakes_in_epoch != 0, mistakes_in_epoch

    def memorize_prior_distribution(self, y_train: List[str]):
        for y in y_train:
            self.apriori_probability[y] += 1

    def test(self, X_test: List[np.ndarray], y_test: List[str]):
        self.calculate_sequence_prediction_error(X_test, y_test)
        self.calculate_char_prediction_error(X_test, y_test)

    def calculate_sequence_score(self, possible_sequence: str, example: np.ndarray):
        score: float = sum([np.dot(self.character_templates[letter], example[:, i]) + \
                            self.biases[letter] for i, letter in enumerate(possible_sequence)]) + \
                       self.apriori_probability[possible_sequence]
        return score

    def calculate_sequence_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        return super(FixedNSequencesLinearClassifier, self).calculate_sequence_prediction_error(X_test, y_test)

    def calculate_char_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        return super(FixedNSequencesLinearClassifier, self).calculate_char_prediction_error(X_test, y_test)

    def classify(self, example: np.ndarray):
        sequence_max_score = float("-inf")
        final_prediction = ""
        for possible_sequence in list(
                set(filter(lambda x: len(x) == example.shape[1], list(self.apriori_probability.keys())))):
            sequence_score = self.calculate_sequence_score(possible_sequence, example)
            if sequence_score > sequence_max_score:
                sequence_max_score = sequence_score
                final_prediction = possible_sequence
        return final_prediction


class ConsecutiveLetterLinearClassifier(LinearClassifier):

    def __init__(self, n_features: int):
        super().__init__(n_features)
        self.apriori_probability = defaultdict(lambda: 0)

    def train(self, X: List[np.ndarray], y: List[str], parameters: TrainingParameters = TrainingParameters()):
        self.memorize_prior_distribution(y)
        return super(ConsecutiveLetterLinearClassifier, self).train(X, y, parameters)

    def train_single_epoch(self, X: List[np.ndarray], y: List[str], epoch_number: int, lr: float) -> Tuple[bool, int]:
        mistakes_in_epoch: int = 0
        for i, (example, true_class) in enumerate(zip(X, y)):
            final_prediction = self.classify(example)
            if not final_prediction == true_class:
                for j, (predicted, real) in enumerate(zip(final_prediction, true_class)):
                    self.character_templates[real] += lr * example[:, j]
                    self.character_templates[predicted] -= lr * example[:, j]
                    self.biases[real] += lr
                    self.biases[predicted] -= lr
                mistakes_in_epoch += 1
        print("There was {} mistakes in epoch {}".format(mistakes_in_epoch, epoch_number))
        return mistakes_in_epoch != 0, mistakes_in_epoch

    def memorize_prior_distribution(self, y_train: List[str]):
        """
        memorizes the distribution of pairs of consecutive characters in the training set
        :param y_train:
        :return:
        """
        for y in y_train:
            for i in range(len(y) - 1):
                self.apriori_probability[str(y[i]) + str(y[i + 1])] += 1

    def test(self, X_test: List[np.ndarray], y_test: List[str]):
        self.calculate_sequence_prediction_error(X_test, y_test)
        self.calculate_char_prediction_error(X_test, y_test)

    def calculate_sequence_score(self, possible_sequence: str, example: np.ndarray):
        score: float = sum([np.dot(self.character_templates[letter], example[:, i]) + \
                            self.biases[letter] for i, letter in enumerate(possible_sequence)]) + \
                       sum([self.apriori_probability[str(possible_sequence[j]) + str(possible_sequence[j + 1])] for j in
                            range(len(possible_sequence) - 1)])
        return score

    def calculate_sequence_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        return super(ConsecutiveLetterLinearClassifier, self).calculate_sequence_prediction_error(X_test, y_test)

    def calculate_char_prediction_error(self, X_test: List[np.ndarray], y_test: List[str]):
        return super(ConsecutiveLetterLinearClassifier, self).calculate_char_prediction_error(X_test, y_test)

    def classify(self, example: np.ndarray):
        already_computed_combinations: Dict[str, float] = {
            letter: np.dot(self.character_templates[letter], example[:, -1]) + \
                    self.biases[letter] for letter in string.ascii_lowercase
        }
        top_combinations: Dict[str, float] = {}
        for i in reversed(range(-1 * example.shape[1], -1)):
            # go backwards through the example from the last letter and compute the best "subword" consisting of the last (-1 * i) letters
            for letter in string.ascii_lowercase:
                all_letter_combinations = {
                    letter + previous_comb: np.dot(self.character_templates[letter], example[:, i]) + \
                                            self.biases[letter] + self.apriori_probability[letter + previous_comb[0]] +
                                            already_computed_combinations[previous_comb]
                    for previous_comb in already_computed_combinations.keys()
                }
                best_combination = max(all_letter_combinations.items(), key=operator.itemgetter(1))[0]
                # pick the best subsequence of letters for a given starting letter and add it to top_combinations, do such for all letters
                top_combinations[best_combination] = all_letter_combinations[best_combination]
            already_computed_combinations = top_combinations
            top_combinations: Dict[str, float] = {}
        best_combination_overall = max(already_computed_combinations.items(), key=operator.itemgetter(1))[0]
        return best_combination_overall

    def get_all_sequences_of_length(self, length: int, alphabet: List[Any]):
        for item in itertools.permutations(alphabet, length):
            yield "".join(item)
