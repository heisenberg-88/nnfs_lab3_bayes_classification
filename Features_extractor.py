import os
import numpy as np
from nltk.corpus import stopwords
from collections import Counter


def extract_features(dir,dictionary,dictionary_size):
    stops = stopwords.words('english')
    stops.append('subject:')
    stops.append('subject :')

    files = [os.path.join(dir, fi) for fi in os.listdir(dir)]
    features_matrix = np.zeros((len(files),dictionary_size))
    labels = np.zeros(len(files))

    file_number = 0
    for file in files:
        # print("file no. "+str(file_number)+" \n")
        with open(file,encoding="utf8",errors='ignore') as mail:
            tempwordsArray = []
            for line in mail:
                tempwords = line.split()
                for word in tempwords:
                    if len(word)>1 and word.isalpha()==True:
                        if word not in stops:
                            tempwordsArray.append(word.lower())
            tempdict = dict(Counter(tempwordsArray))

            index = 0
            for dictword in dictionary:
                # print("searching "+dictword+" in tempdict")
                if dictword in tempdict:
                    # print("found..")
                    features_matrix[file_number][index] = tempdict[dictword]
                index+=1



        filepathwords = file.split('/')
        lastwordfrompath = filepathwords[len(filepathwords) - 1]
        if lastwordfrompath.startswith("spmsg"):
            labels[file_number] = 1

        file_number += 1


    return features_matrix , labels







