import xml.etree.cElementTree as ET
import Input
import os


def outputResult(id, type, lang="en", prepath=""):
    root = ET.Element("author", id=id, lang=lang, type=str(type))
    tree = ET.ElementTree(root)
    tree.write("{}{}.xml".format(prepath, id))


def outputScores(id, feed, scores, label, path):
    with open(path, "w") as file:
        file.write(f"tweet\tscore\tlabel\n")
        for tweet, score in zip(feed, scores):
            file.write(f"{tweet}\t{score}\t{label} \n")


if __name__ == "__main__":
    user = Input.getUsers("../data/en/")[0]
    outputResult(user, "1", lang="en", prepath="t_output/")
