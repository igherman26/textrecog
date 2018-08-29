from charClassifier import Classifier
from textDetection import loadAndRotateImage, extractWordsROIs, extractLettersROIs
from pprint import pprint
import sys


def main():
    if len(sys.argv) != 2:
        print ("usage: {} <file_path>".format(sys.argv[0]))
        return

    text = []
    charClassifier = Classifier() 

    # rotate the image if necessary
    rotatedImage = loadAndRotateImage(sys.argv[1], testing=False)
    
    # obtain words regions of interest
    wordsROIs = extractWordsROIs(rotatedImage, testing=False, showRectangle=False, saveWords=True)

    if wordsROIs is None or len(wordsROIs) == 0:
        return

    # process words
    for wordROI in wordsROIs:
        word = ""

        # obtain letters
        lettersROIs = extractLettersROIs(wordROI, testing=False, showRectangle=False, saveLetters=True)

        # classify the letters 
        for letter in lettersROIs:
            word += charClassifier.classify(letter)

        text.append(word)

    pprint(text)

    # print no. of words and letters
    print ("There are {} words.".format(len(wordsROIs)))

if __name__ == "__main__":
    main()