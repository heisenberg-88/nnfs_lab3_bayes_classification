import os
from nltk.corpus import stopwords
from collections import Counter

def create_dictionary(dir,dictionary_size):
    stops = stopwords.words('english')
    stops.append('subject:')
    stops.append('subject :')

    email_file_names = [os.path.join(dir,f) for f in os.listdir(dir)]
    wordlist = []
    for mail in email_file_names:
        with open(mail,encoding="utf8",errors='ignore') as m:
            for line in m:
                words = line.split()
                for word in words:
                    if len(word)>1 and word.isalpha()==True:
                        if word not in stops:
                            wordlist.append(word.lower())


    data = Counter(wordlist)
    print("total words: "+str(len(data)))
    data =  dict(data.most_common((dictionary_size)))
    return data



