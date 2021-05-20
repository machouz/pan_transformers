from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import glob
import os
from nltk.tokenize import TweetTokenizer
import pandas as pd
import emoji


tt = TweetTokenizer()

wordsSubstitute = {"#USER#": "@USER",
                   ",": " ",
                   "#HASHTAG#": "",
                   "#URL#": "HTTPURL",
                   "RT : ": "",
                   "RT @": "@",
                   "RT #": "#"}


def getFeedForUser(id, prepath=""):
    tree = ET.parse("{}{}.xml".format(prepath, id))
    root = tree.getroot()
    return [elem.text for elem in root.iter("document")]


def getBaseNameWithoutExt(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def getUsers(path="data/en/"):
    files = glob.glob(path + "*.xml")
    return [getBaseNameWithoutExt(file) for file in files]


def getUsersFeed(users, path):
    feeds = [getFeedForUser(id, path) for id in users]
    return dict(zip(users, feeds))


def getUsersLabel(path="data/en/"):
    user_label = {}
    with open(path + "truth.txt") as file:
        for line in file:
            items = line.rsplit(":::")
            id, label = items[0], items[1][0]
            user_label[id] = label
    return user_label


def removeStopWords(text):
    word_list = text.split()
    filtered_words = [
        word for word in word_list if word not in englishStopWords]

    return " ".join(filtered_words)


# Preprocess text (username and link placeholders)
def preprocess(text):
    for key, word in wordsSubstitute.items():
        text = text.replace(key, word)
    # text = emoji.demojize(text)
    # text = " ".join(tt.tokenize(text))
    # text = text.lower()
    # text = removeStopWords(text)
    return text


def preprocessFeed(feed):
    return [preprocess(text) for text in feed]


def preprocessAndJoin(feed):
    return " . ".join(preprocessFeed(feed))


def unnest(df, col, reset_index=False):
    import pandas as pd
    col_flat = pd.DataFrame([[i, x]
                             for i, y in df[col].apply(list).iteritems()
                             for x in y], columns=['I', col])
    col_flat = col_flat.set_index('I')
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)
    if reset_index:
        df = df.reset_index(drop=True)
    return df


def get_frame(path):
    users = getUsers(path)

    users_label = getUsersLabel(path)
    users_feed = getUsersFeed(users, path)
    users_preprocessedFeed = {
        key: preprocessFeed(value) for key, value in users_feed.items()
    }
    frame = pd.DataFrame(users_preprocessedFeed.items(),
                         columns=['id', 'text'])
    frame['label'] = frame['id'].map(users_label)

    return frame


def get_data_dict(path, with_label=True):
    users = getUsers(path)
    users_feed = getUsersFeed(users, path)

    if with_label:
        users_label = getUsersLabel(path)

    if with_label:
        return [
            {
                "id": key,
                "input": preprocessFeed(value),
                "label": int(users_label[key]) if users_label[key].isdigit() else users_label[key]
            } for key, value in users_feed.items()
        ]
    else:
        return [
            {
                "id": key,
                "input": preprocessFeed(value),
            } for key, value in users_feed.items()
        ]


if __name__ == "__main__":
    path = "data/en/"
    users = getUsers(path)

    users_label = getUsersLabel(path)
    users_feed = getUsersFeed(users, path)
    users_preprocessedFeed = {
        key: preprocessFeed(value) for key, value in users_feed.items()
    }
    users_preprocessedAndJoinedFeed = {
        key: preprocessAndJoin(value) for key, value in users_feed.items()
    }

    print("Tweet: ", users_feed[users[0]][0])
    print("Preprocessed tweet: ", users_preprocessedFeed[users[0]][0])

    print("Preprocessed joined tweet: ",
          users_preprocessedAndJoinedFeed[users[0]])
    print("Preprocessed joined tweet: ", len(
        users_preprocessedAndJoinedFeed[users[0]]))
    print("Label: ", users_label[users[0]])
    frame = pd.read_csv(
        'summary/custom_hate.tsv',
        index_col=None,
        sep="\t",
    )
    frame['gold'] = frame['id'].map(users_label)
    frame.to_csv("summary/custom_hate.tsv", sep="\t", index=False)
    print(frame.head())
