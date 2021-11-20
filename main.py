import itertools
import string
import time

from classifiers.linearclassifiers import BasicLinearClassifier, FixedNSequencesLinearClassifier, \
    ConsecutiveLetterLinearClassifier
from datautils.helpers import load_examples
from datautils.models import TrainingParameters

if __name__ == '__main__':
    trn_X, trn_Y, trn_img = load_examples('ocr_names_images/trn')
    tst_X, tst_Y, tst_img = load_examples('ocr_names_images/tst')

    print("------------------ Task 2.1 -----------------------------")
    clf = BasicLinearClassifier(n_features = 8256)
    clf.train(trn_X, trn_Y, parameters=TrainingParameters(LearningRate=1))
    clf.test(tst_X, tst_Y)
    print("------------------ End of Task 2.1 -----------------------------")


    print("-------------------- Task 2.2 ----------------------------")
    clf = ConsecutiveLetterLinearClassifier(n_features=8256)
    clf.train(trn_X, trn_Y, parameters=TrainingParameters(LearningRate=1))
    clf.test(tst_X, tst_Y)
    print("------------------ End of Task 2.2 -----------------------------")


    print("-------------------- Task 2.3 ----------------------------")
    clf = FixedNSequencesLinearClassifier(n_features = 8256)
    clf.train(trn_X,trn_Y, parameters=TrainingParameters(LearningRate=1))
    clf.test(tst_X, tst_Y)
    print("------------------ End of Task 2.3 -----------------------------")


