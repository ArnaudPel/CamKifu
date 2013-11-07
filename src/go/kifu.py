__author__ = 'Kohistan'


class Tree:
    """ A tree structure modeling the moves of the game.
    >>> kifu = Kifu()
    >>> kifu.move("B[aa]")
    >>> kifu.move("W[ba]")
    >>> kifu.move("ca")
    >>> kifu.undo()
    >>> kifu.move("cb")
    >>> kifu.move("db")
    >>> kifu.move("eb")
    >>> kifu.undo()
    >>> kifu.undo()
    >>> kifu.undo()
    >>> kifu.move("cb")
    >>> kifu.move("dc")
    >>> print(kifu)
    root: B[aa]
    B[aa]: W[ba]
    W[ba]: ca cb
    ca:
    cb: db dc
    db: eb
    eb:
    dc:
    <BLANKLINE>
     """

    def __init__(self, parent=None, move="root"):
        self.parent = parent
        self.value = move
        self.children = []

    def move(self, move):
        for child in self.children:
            if child.value == move:
                return child
        child = Tree(self, move)
        self.children.append(child)
        return child

    def __repr__(self):
        stri = self.value + ":"
        for child in self.children:
            stri += " " + child.value
        stri += "\n"
        for child in self.children:
            stri += repr(child)
        return stri


class Kifu:
    """ A game record. """
    def __init__(self):
        self.root = Tree()
        self.current = self.root

    def move(self, move):
        self.current = self.current.move(move)

    def undo(self):
        self.current = self.current.parent

    def __repr__(self):
        return repr(self.root)

    @staticmethod
    def parse(sgf):
        print "parsing " + sgf
        kifu = Kifu()
        with open(sgf) as mfile:
            for line in mfile:
                length = len(line)
                i = 0
                while i < length:
                    if line[i] == ';':
                        i += 1
                        keyword = ""
                        while (i < length) and (line[i] != "["):
                            keyword += line[i]
                            i += 1
                        if keyword == 'B' or keyword == 'W':
                            kifu.move(keyword + line[i:i+4])
                    else:
                        i += 1
            kifu.current = kifu.root
        return kifu

if __name__ == '__main__':
    import doctest
    doctest.testmod()