Code of Project : 
-----------------------
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Data Set Path
path = "C:\\Users\\DELL\\Documents\\Automatic Language Identification Useing K-Means Clustring AI Project\\DataSets\\dataset.csv"
languages = pd.read_csv(path)
# languages.head(10)
languages.tail(10)

lang_labels = languages['language'].unique()
print(lang_labels)

# ['Estonian' 'Swedish' 'Thai' 'Tamil' 'Dutch' 'Japanese' 'Turkish' 'Latin'
#  'Urdu' 'Indonesian' 'Portugese' 'French' 'Chinese' 'Korean' 'Hindi'
#  'Spanish' 'Pushto' 'Persian' 'Romanian' 'Russian' 'English' 'Arabic']

languages_labels = []
temp_list = []
Matrix = []
Matrix = list(languages)
# i = 0
print(Matrix)
print(type(Matrix))
print(languages.iloc[0:5, 1:])
# for label in lang_labels:
label = 'English'
raw = -1

# Labels = np.array(languages.iloc[:, 1:])
# Sequences = np.array(languages.iloc[:, 0:1])
Labels = np.array(languages.iloc[:, 1:]) # 22 Languages
Sequences = np.array(languages.iloc[:, 0:1]) # Sentences
print (len(Labels))
print (len(Sequences))

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
for lang in lang_labels:
    i = -1
    f_en = open(lang + ".txt", "w", encoding='utf-8')
    for label in Labels:
        i +=1
        if label == lang:
            seq = str(Sequences[i])
#             print(type(seq))
            f_en.write(seq)
    f_en.close()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # clearing
languages_texts = []

for lang in lang_labels:
    i = -1
    f = open(lang + ".txt", "r", encoding='utf-8')
    text = f.read()[1:]
    text = text.replace('[', '').replace(']', '').replace('"', '').replace('\'', '').replace('–', '').split()
    languages_texts.append(text)
    f.close()
    
print(languages_texts[21])

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Calculating bag of words
# word_set = set(l_A).union(set(l_B)).union(set(l_C))

word_set = set()
l=0
for index in range(len(lang_labels)):
    l +=1
    word_set = word_set.union(set(languages_texts[index]))
    
set_ = list(word_set)
# print(len(set_))

word_dict_languages = [] 
word_dict_lang = {}

for label in range(len(lang_labels)):
    word_dict_lang = dict.fromkeys(word_set, 0)
    word_dict_languages.append(word_dict_lang)

print(len(word_dict_languages[21]))
word_dict_languages[21]['كليب'] +=1
print(word_dict_languages[21]['كليب'])

for label in range(len(lang_labels)):
    for seq in languages_texts[label]:
#         for word in range(len(languages_texts[label][seq])):
                #Lang #Seq #Word
        word_dict_languages[label][seq] += 1
        
# word_ = list(word_dict_languages[21].values())
print(word_dict_languages[20]['المصدر'])

def compute_tf(word_dict, language):
    tf = {}
    sum_langkeys = len(language)
    for word, count in word_dict.items():
        tf[word] = count/sum_langkeys
    return tf

# List Of Dict
tf_languages = []
for label in range(len(lang_labels)):
    tf_languages.append(compute_tf(word_dict_languages[label], languages_texts[label]))

print(tf_languages[21]['كليب'])
def compute_idf(strings_list):
    n = len(strings_list)
    idf = dict.fromkeys(strings_list[0].keys(), 0)
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1
    
    for word, v in idf.items():
        idf[word] = np.log(n / float(v))
    return idf
    
idf = compute_idf(tf_languages)
idf

def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf

tf_idf_Languages = []
for label in range(len(lang_labels)):
    tf_idf_Languages.append(compute_tf_idf(tf_languages[label], idf))

print(tf_idf_Languages[21]['كليب'])

def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]", " ", line)
#     print(line)
    return line
    
final = []
f_lang = []
for label in range(len(lang_labels)):
    for seq in languages_texts[label]:
        f_lang.append(preprocessing(seq))
    final.append(f_lang)
    
vectorizer = TfidfVectorizer(stop_words='english')

stringList = []
string = ""
for label in range(len(lang_labels)):
    for seq in languages_texts[label]:
        string+=seq
    stringList.append(string)
    string = ""
s = stringList[:5]
print(s)
X = vectorizer.fit_transform(stringList)

model = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=100,
       n_clusters=22, n_init="random", n_jobs=None, precompute_distances='auto',
       random_state=42, tol=0.0001, verbose=0)

terms = vectorizer.get_feature_names()
terms

# print(type(true_k))
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :20]:
        print(' %s' % terms[ind]),
    print('------------------------------------------')

print("\n")
print("Prediction")


