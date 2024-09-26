from argparse import ArgumentParser
import os
import struct
import sys

def is_binary_format(filename):
  try:
    with open(filename, "rb") as f:
      chunk = f.read(1024)
    chunk.decode("utf-8")
  except UnicodeDecodeError:
    return True, 0
  except Exception as e:
    print("Fail to open file:", filename, e)
    return False, 1
  return False, 0

def read_binary_file_as_uint8(filename):
  try:
    with open(filename, "rb") as f:
      binary_data = f.read()
      uint8_data = struct.unpack(f"{len(binary_data)}B", binary_data)
      return uint8_data
  except Exception:
    return None

def main(args):
  filename = args.file
  print(f"Target file is:", filename)
  is_binary, retcode = is_binary_format(filename)
  if not is_binary:
    sys.exit(retcode)
  bytestream = read_binary_file_as_uint8(filename)
  if bytestream is None:
    print("Fail to read binary file as uint8")
    sys.exit(1)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-f", "--file", type=str, required=True)
  args = parser.parse_args()
  main(args)
