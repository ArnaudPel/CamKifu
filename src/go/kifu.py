from config.guiconf import gsize
from go.sgf import Collection, GameTree, Node, Parser
from go.sgfwarning import SgfWarning

__author__ = 'Kohistan'


class Kifu:
    """
    Utility class simplifying common interactions with the SGF structure.
    Will become more complicated if variations support is introduced.

    self.game -- the GameTree object backing the recording of this kifu.

    """
    def __init__(self, game, sgffile=None):
        self.game = game
        self.sgffile = sgffile

    def append(self, move):
        node = Node(self.game, self.game.nodes[-1])
        r, c = move.getab()
        node.properties[move.color] = [r + c]  # sgf properties are in a list
        node.number()
        self.game.nodes.append(node)

    def pop(self):
        self.game.nodes.pop()

    def last_move(self):
        """
        Note: this is a naive implementation based on the assumption that the game has no children.
        In other words, that there are not variations at all.

        """
        return self.game.nodes[-1].getmove()

    def next_color(self):
        current = self.last_move()
        if current is not None:
            return 'B' if current.color == 'W' else 'W'
        else:
            return 'B'

    def relocate(self, origin, dest):
        """
        Note: this is a naive implementation based on the assumption that the game has no children.
        In other words, that there are not variations at all.

        """
        for node in self.game.nodes:
            mv = node.getmove()
            if mv and mv.x == origin.x and mv.y == origin.y:
                a, b = dest.getab()
                node.properties[mv.color] = [a+b]

    def save(self):
        if self.sgffile is not None:
            with open(self.sgffile, 'w') as f:
                self.game.output(f)
                print "Game saved to: " + self.sgffile
        else:
            raise SgfWarning("No file defined, can't save.")

    def __repr__(self):
        return repr(self.game)

    @staticmethod
    def new():
        """
        Create an empty Kifu.

        """
        # initialize game
        collection = Collection()
        game = GameTree(collection)
        collection.children.append(game)

        # add context node
        context = Node(game, None)
        context.properties["SZ"] = [gsize]
        context.properties['C'] = ["Recorded with Camkifu."]
        context.number()
        game.nodes.append(context)

        return Kifu(game)

    @staticmethod
    def parse(filepath):
        """
        Create a Kifu reflecting the given file.

        """
        parser = Parser()
        f = file(filepath)
        sgf_string = f.read()
        f.close()
        collection = Collection(parser)
        parser.parse(sgf_string)
        return Kifu(collection[0])


if __name__ == '__main__':
    colors = ['B', 'W']
    kifu = Kifu.new()
    previous = kifu.game.nodes[-1]
    for i in range(gsize):
        nod = Node(kifu.game, previous)
        nod.properties[colors[i % 2]] = [chr(i+97)+chr(i+97)]
        kifu.game.nodes.append(nod)
        previous = nod

    f_out = file("/Users/Kohistan/Documents/go/Perso Games/updated.sgf", 'w')
    kifu.game.parent.output(f_out)

















