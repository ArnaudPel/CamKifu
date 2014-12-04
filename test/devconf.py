
__author__ = 'Arnaud Peloquin'

"""
Configuration of things that should not make it to "production" releases.

"""

gobanloc_npz = "../res/temp/gobanlocs/"

# width of the screen, pixels
screenw = 1920

# height of the screen, pixels
screenh = 1200

# location of the board_finder window. set to None to center
bf_loc = (1100, 70)

# location of the stones_finder window. set to None to center
sf_loc = (1100, 600)

# The moves the Dummy Finder should find :)
dummy_sf_args = ("kgs",
                ["W[H8]", "B[J8]", "W[K12]", "B[F12]", "W[F11]", "B[H10]",
                 "W[J14]", "B[J12]", "W[J11]", "B[J13]", "W[K13]"]
)

