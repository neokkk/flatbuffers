import sys

if sys.byteorder == "little":
    print("little endian")
else:
    print("big endian")
