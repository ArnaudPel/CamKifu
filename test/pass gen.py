import random

__author__ = 'Kohistan'

def passgen():
    global password, rand, i, value, c
    password = ""
    rand = random.Random()
    for i in range(10):
        value = rand._randbelow(26)
        c = chr(value + 97)
        password += c
    print password

def casefixer(path):
    # how to list files in a folder ??
    pass


if __name__ == '__main__':
    passgen()
