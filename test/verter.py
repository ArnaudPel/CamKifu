__author__ = 'Kohistan'



if __name__ == '__main__' :
    file = open("/Users/Kohistan/PycharmProjects/CamKifu/res/brit-a-z.txt")
    words = []
    for line in file:
        if line.__contains__("verter"):
            words.append(line)
    file.close()
    print(words)