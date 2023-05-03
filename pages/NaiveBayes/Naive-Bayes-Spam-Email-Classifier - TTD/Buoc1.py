from os import walk
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

EXAMPLE_FILE = 'SpamData/01_Processing/practice_email.txt'
SPAM_1_PATH = 'SpamData/01_Processing/spam_assassin_corpus/spam_1'
SPAM_2_PATH = 'SpamData/01_Processing/spam_assassin_corpus/spam_2'
EASY_NONSPAM_1_PATH = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_1'
EASY_NONSPAM_2_PATH = 'SpamData/01_Processing/spam_assassin_corpus/easy_ham_2'
SPAM_CAT = 1
HAM_CAT = 0
VOCAB_SIZE = 2500
DATA_JSON_FILE = 'SpamData/01_Processing/email-text-data.json'
WORD_ID_FILE = 'SpamData/01_Processing/word-by-id.csv'
TRAINING_DATA_FILE = 'SpamData/02_Training/train-data.txt'
TEST_DATA_FILE = 'SpamData/02_Training/test-data.txt'
THUMBS_UP_FILE = 'SpamData/01_Processing/wordcloud_resources/thumbs-up.png'
THUMBS_DOWN_FILE = 'SpamData/01_Processing/wordcloud_resources/thumbs-down.png'
CUSTOM_FONT_FILE = 'SpamData/01_Processing/wordcloud_resources/OpenSansCondensed-Bold.ttf'

# Giai đoạn 1 - Download data
'''
stream = open(EXAMPLE_FILE, encoding='latin-1')
is_body = False
lines = []

for line in stream: 
    if is_body:
        lines.append(line)
    elif line =='\n':
        is_body = True
stream.close()
email_body = '\n'.join(lines)
print(email_body)
'''

def generate_squares(N):
    for my_number in range(N):
        yield my_number ** 2

'''
for i in generate_squares(10):
    print(i, end = ' -> ')
'''

def email_body_generator(path): 
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            filepath = join(root, file_name)
            stream = open(filepath, encoding='latin-1')
            is_body = False
            lines = []
            for line in stream: 
                if is_body:
                    lines.append(line)
                elif line =='\n':
                    is_body = True
            stream.close()
            email_body = '\n'.join(lines)
            yield file_name, email_body

def df_from_directory(path, classification):
    rows = []
    row_names = []
    
    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
        
    return pd.DataFrame(rows, index = row_names)

spam_emails = df_from_directory(SPAM_1_PATH, SPAM_CAT)
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, SPAM_CAT))

# Print 5 the first rows của Spam
'''
print(spam_emails.head())
print(spam_emails.shape)
'''

ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = ham_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT))

# Print 5 the first rows của Ham
print(ham_emails.head())
print(ham_emails.shape)

# Nối 2 tập spam và ham thành một tập data
data = pd.concat([spam_emails, ham_emails])
print('Shape of entire dataframe is ', data.shape)

# In 5 hàng đầu tiên
print(data.head())
# In 5 hàng cuối cùng
print(data.tail())

# Giai đoạn 2: Làm sạch dữ liệu






