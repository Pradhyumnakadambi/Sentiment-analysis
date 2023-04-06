# SENTIMENT ANALYSIS WITH HOTEL REVIEWS

import pandas as pd
import nltk
from nltk import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import string
import re
import tkinter
from tkinter import messagebox
import threading

# Following packages required for the analysis
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')


# Creation of interface window
Root = tkinter.Tk()
Root.configure(bg='black')
Root.resizable(False, False)
frame = tkinter.Frame(Root)
frame.pack()
Root.geometry("550x350")

df = pd.read_csv('tripadvisor_hotel_reviews.csv', header=None)


def test():
    global Root, df, dup
    # Removing punctuations
    clnd_revs = []
    dup = df
    for i in range(len(df)):
        no_punc = [revs for revs in df[0][i] if revs not in string.punctuation]
        st = ''.join(no_punc)
        clnd_revs.append(st)

    # Tokenizing
    for i in range(len(df)):
        clnd_revs[i] = re.split("\W+", clnd_revs[i])

    # Stop words
    stopword = nltk.corpus.stopwords.words('english')
    for i in range(len(df)):
        clnd_revs[i] = [word for word in clnd_revs[i] if word not in stopword]

    # Lemmatize
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    lemmatizer = WordNetLemmatizer()
    for i in range(len(clnd_revs)):
        sent = ' '.join(clnd_revs[i])
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        dup[0][i] = clnd_revs[i]
        clnd_revs[i] = " ".join(lemmatized_sentence)
        df[0][i] = clnd_revs[i]
    print(dup.head(6))
    return df


def cleaning():
    global Root
    test()
    global df
    if len(df[0]) == len(df[1]):
        print("Dataset has equal number of records.")
        cnt1 = 0
        cnt2 = 0
        for i in range(len(df)):
            if df[0][i] == "" or df[0][i] == " ":
                cnt1 += 1
            elif df[1][i] == "" or df[1][i] == " ":
                cnt2 += 1
        if cnt1 == 0 and cnt2 == 0:
            print("No incomplete data present")

        # Checking for invalid data
        arr = []
        for i in range(1, len(df)):
            if int(df[1][i]) < 1 or int(df[1][i]) > 5:
                arr.append[i]
        if len(arr) == 0:
            print("No invalid data present")
        else:
            print("Invalid data present.")

        msg = "Data cleaning complete."
        print(msg)

    messagebox.showinfo("Complete", msg)
    But1.destroy()
    menu1()


def analysis():
    global df
    # test()
    # cleaning()
    reviews = df.iloc[:, 0].values
    hotel_reviews = pd.DataFrame(reviews)
    nltk.download('vader_lexicon', quiet="True")
    vader = SentimentIntensityAnalyzer()
    func = lambda title: vader.polarity_scores(title)['compound']
    hotel_reviews['Polarity_score'] = hotel_reviews[0].apply(func)

    def analysis_score(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    hotel_reviews['sentiment'] = hotel_reviews['Polarity_score'].apply(analysis_score)
    pos = 0
    neg = 0
    neu = 0
    for i in range(1, len(hotel_reviews)):
        if hotel_reviews['Polarity_score'][i] > 0:
            pos += 1
        elif hotel_reviews['Polarity_score'][i] < 0:
            neg += 1
        else:
            neu += 1
    print(f"Positive reviews: {pos} \nNegative reviews: {neg} \nNeutral reviews: {neu}")

    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    hotel_reviews['sentiment'].value_counts().plot(kind='bar')
    plt.show()

    hotel_reviews['sentiment'].value_counts().plot(kind='pie', autopct='%1.0f%%', fontsize=12, figsize=(9, 6),
                                                   colors=["blue", "red", "yellow"])
    plt.ylabel("Hotel Reviews Sentiment", size=14)
    plt.show()

    return df


# Table structure - width of columns
def indent(table, wd):
    temp = 0
    sz = []
    sz1 = []
    while temp < wd:
        for row in table:
            for elm in range(len(row)):
                if elm == temp:
                    sz.append(len(str(row[temp])))
                else:
                    continue
        if sz:
            sz1.append(max(sz))
        temp += 1
        sz = []
    return sz1


# Table structure - Main
# Displaying record(s)
class tab:
    def __init__(self, table):
        self.table = table
        self.root = tkinter.Tk()
        rows = 2
        col1 = len(self.table[0])
        col2 = len(self.table[1])
        if col1 > col2:
            columns = col1
        else:
            columns = col2
        columns, rows = rows, columns
        wid = indent(self.table, columns)
        ret = tkinter.Button(self.root, text="Close", command=self.root.destroy)
        ret.grid(row=rows + 1, column=0, sticky=tkinter.W)
        for i in range(rows):
            for j in range(columns):
                try:
                    if i == 0:
                        selfent = tkinter.Entry(self.root, width=wid[j], fg='black',
                                                font=('Ariel', 16, "bold"))
                    else:
                        selfent = tkinter.Entry(self.root, width=wid[j], fg='black',
                                                font=('Calibre(body)', 16))
                    selfent.grid(row=i, column=j)
                    selfent.insert(tkinter.END, table[j][i])
                except IndexError:
                    continue

        self.root.mainloop()

    def destroy(self):
        self.root.destroy()


def word_sent(test_subset):
    global tble
    sia = SentimentIntensityAnalyzer()
    pos_words = []
    neu_words = []
    neg_words = []
    lst = test_subset.split(" ")

    for word in range(len(lst)):
        print(lst[word])
        sentc = lst[word]
        sc = sia.polarity_scores(sentc)
        if sc['compound'] > 0:
            pos_words.append(sentc)
        elif sc['compound'] < 0:
            neg_words.append(sentc)
        elif sc['compound'] == 0:
            neu_words.append(sentc)

    # print('Positive :', pos_words)
    # print('Neutral :', neu_words)
    # print('Negative :', neg_words)
    pos_words.insert(0, "Positive")
    neg_words.insert(0, "Negative")
    tble = [pos_words, neg_words]
    tab(tble)


def refresh(*argv):
    for i in argv:
        i.destroy()


def inp():
    global df, root_inp, tble
    reviews = df.iloc[:, 0].values
    hotel_reviews = pd.DataFrame(reviews)
    vader = SentimentIntensityAnalyzer()
    function = lambda title: vader.polarity_scores(title)['compound']
    hotel_reviews['Polarity_score'] = hotel_reviews[0].apply(function)
    root_inp = tkinter.Tk()
    frm = tkinter.Frame(root_inp)
    frm.pack()
    root_inp.geometry("400x250")
    root_inp.resizable(False, False)
    lb = tkinter.Label(root_inp, text="Customer number", font=('calibre', 10, 'bold'))
    ent = tkinter.Entry(root_inp)

    def but():
        global scr, lb1, lb2, root_inp
        cust = ent.get()
        cust = int(cust)
        print(cust)
        for i in range(len(df[0])):
            if i == cust:
                scr = hotel_reviews['Polarity_score'][i]
                lb1 = tkinter.Label(root_inp, text=f"Polarity score: {scr}", font=('calibre', 15))
                if scr > 0:
                    lb2 = tkinter.Label(root_inp, text="Positive reviews given by this customer.", font=('calibre', 15))
                elif scr < 0:
                    lb2 = tkinter.Label(root_inp, text="Negative reviews given by this customer.", font=('calibre', 15))
                else:
                    lb2 = tkinter.Label(root_inp, text="Neutral reviews given by this customer.", font=('calibre', 15))
                lb1.place(x=5, y=110)
                lb2.place(x=5, y=140)
                break
        word_sent(dup[0][i])

    bt = tkinter.Button(root_inp, text="Submit", command=but, font=('calibre', 10),
                        height=1, width=10)
    bt1 = tkinter.Button(root_inp, text="Refresh", command=lambda: refresh(lb1, lb2), font=('calibre', 10), height=1,
                         width=10)
    lb.place(x=5, y=10)
    ent.place(x=130, y=10)
    bt.place(x=10, y=60)
    bt1.place(x=165, y=60)

    root_inp.mainloop()


def menu1():
    but1 = tkinter.Button(Root, text="Proceed to input", command=threading.Thread(target=inp).start,
                          font=('calibre', 10), height=2, width=15)
    but2 = tkinter.Button(Root, text="View analysis", command=threading.Thread(target=analysis).start,
                          font=('calibre', 10), height=2, width=15)
    but1.place(x=105, y=200)
    but2.place(x=310, y=200)


def menu():
    global Root, But1
    hd = tkinter.Label(Root, text="Sentiment Analysis", font=('calibre', 15, 'bold'))
    tp = tkinter.Label(Root, text="Hotel Reviews", font=('calibre', 12, 'italic'))
    Root.configure(background='black')
    hd.pack()
    tp.pack(pady=40)
    But1 = tkinter.Button(Root, text="Start data cleaning", command=threading.Thread(target=cleaning).start,
                          font=('calibre', 10), height=2, width=14)
    But1.pack(pady=80)
    Root.mainloop()


menu()
