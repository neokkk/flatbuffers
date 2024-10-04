from argparse import ArgumentParser
from enum import Enum
import numpy as np
import os
import struct
import sys

type_to_struct_format = {
  int: "i",
  float: "f",
  bool: "?",
  bytes: "s",
  "int8": "b",
  "uint8": "B",
  "int16": "h",
  "uint16": "H",
  "int32": "i",
  "uint32": "I",
  "int64": "q",
  "uint64": "Q",
  "float32": "f",
  "float64": "d",
}

def is_binary_format(filename):
  try:
    with open(filename, "rb") as f:
      chunk = f.read(1024)
    chunk.decode("utf-8")
  except UnicodeDecodeError:
    return True, 0
  except Exception as e:
    print(f"Fail to open {filename}: {e}")
    return False, 1
  print("This file is not encoded to binary format")
  return False, 0

def read_binary_file_as_uint8(filename):
  try:
    with open(filename, "rb") as f:
      binary_data = f.read()
      uint8_data = struct.unpack(f"{len(binary_data)}B", binary_data)
      return uint8_data
  except Exception:
    return None

def unpack(t, s):
  format_char = type_to_struct_format.get(t, None)
  if not format_char:
    raise Exception(f"Fail to unpack {t} type as struct format")
  return struct.unpack(f"{endian}{format_char}", bytes(s))[0]

def has_decimal_point(num):
  return num % 1 != 0

def get_order_indices(l, order="asc"):
  reverse = order == "desc"
  return sorted(range(len(l)), key=lambda x: l[x], reverse=reverse)

def signed_to_unsigned(v, e):
  base = 2 ** e
  return (v + base) % base

def unsigned_to_signed(v, e):
  base = 2 ** (e - 1)
  if v >= base:
    return v - 2 ** e
  return v

def is_bool_acceptable(s):
  _, unsigned, val = is_byte_acceptable(s)
  if unsigned and (val == 0 or val == 1):
    return True, val
  return False, None

def is_value_acceptable(s, t, l):
  unsigned_type = f"u{t}"
  signed = unsigned = True
  val = None
  try:
    val = unpack(t, s)
    if len(s) != l:
      raise Exception
  except Exception:
    print(f"{t} unacceptable")
    signed = False
  try:
    val = unpack(unsigned_type, s)
  except Exception:
    print(f"{unsigned_type} unacceptable")
    unsigned = False
  return signed, unsigned, val

def is_byte_acceptable(s):
  return is_value_acceptable(s, "int8", 1)

def is_short_acceptable(s):
  return is_value_acceptable(s, "int16", 2)

def is_int_acceptable(s):
  return is_value_acceptable(s, "int32", 4)

def is_long_acceptable(s):
  return is_value_acceptable(s, "int64", 8)

def is_string_acceptable(s):
  if all(e == 0 for e in s):
    return True, ""
  if len(s) == 0:
    return False, None
  try:
    print(bytes(s))
    unpacked = struct.unpack(f"{endian}{len(s)}s", bytes(s))
    decoded = unpacked[0].decode("utf-8")
    print(decoded)
    if not decoded.strip():
      raise Exception
    return True, decoded
  except:
    print("Fail to unpack string")
    return False, None

class Value(object):
  def __init__(self, type=None, sub_type=None, numeric_type=None, value=None):
    self.type = self.Type.NONE if type is None else type
    self.sub_type = self.SubType.NONE if sub_type is None else sub_type
    self.numeric_type = self.NumericType.NONE if numeric_type is None else numeric_type
    self.value = value = value

  def __str__(self):
    t = self.type if self.type is not self.Type.NONE else ""
    st = self.sub_type if self.sub_type is not self.SubType.NONE else ""
    nt = self.numeric_type if self.numeric_type is not self.NumericType.NONE else ""
    return f"{t} {st} {nt}: {self.value}"

  class Type(Enum):
    NONE = -1
    UNKNOWN = 0
    PRIMITIVE = 1 # bool, enum, numeric, struct
    OFFSET = 2 # string, vector

  class SubType(Enum):
    NONE = -1
    UNKNOWN = 0
    BOOL = 1
    # ENUM = 2
    NUMERIC = 3
    STRUCT = 4
    STRING = 5
    VECTOR = 6
    TABLE = 7

  class NumericType(Enum):
    NONE = -1
    UNKNOWN = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    UINT8 = 11
    UINT16 = 12
    UINT32 = 13
    UINT64 = 14
    FLOAT32 = 21
    FLOAT64 = 22

class Field(object):
  def __init__(self, table, id, start, end):
    print(f"Field {id} created")
    self.table_ = table
    self.id_ = id
    self.start_ = start
    self.end_ = end
    self.values = []

  def __str__(self):
    v = "["
    for value in self.values:
      v = f"{v}\n\t\t{value}"
    return f"[Field {self.id_}]\n\toffset: {self.start_} ~ {self.end_}\n\tvalues:\t\t{v}\n]"

class Table(object):
  def __init__(self, parser, start=-1, root=False):
    self.parser_ = parser
    self.is_root_ = root
    self.start_ = start if start > -1 else self.parser_.cur_
    self.end_ = -1
    self.parser_.cur_ = self.start_
    self.fields = []
    self.deferred = [] # to check range of field
    print(f"start: {self.start_}")

  def __str__(self):
    v = ""
    for field in self.fields:
      v = f"{v}\n{field}"
    return v

  def parse(self):
    try:
      byte_slice = self.parser_.bytestream_[self.start_:self.start_ + 4] # field size: 4B
      self.vtable_offset_ = unpack("uint32", byte_slice)
      if not self.parser_.mark(self.start_, 4):
        print("Fail to mark to be visited")
        return False
      print(f"vtable_offset: {self.vtable_offset_}")
      if not self.parse_meta():
        print("Fail to parse vtable format")
        return False
      self.parse_fields()
      self.parse_deferred() # second-chance to parse
      return True
    except Exception:
      return False

  def parse_meta(self):
    print("parse metadata")
    vtable_start = self.start_ - self.vtable_offset_
    print(f"vtable_start: {vtable_start}")
    byte_slice = self.parser_.bytestream_[vtable_start:vtable_start + 2] # vtable offset size: 2B
    vtable_size = unpack("uint16", byte_slice)
    print(f"vtable_size: {vtable_size}")
    if not self.parser_.mark(vtable_start, 2):
      print("Fail to mark to be visited: vtable_start")
      return False
    byte_slice = self.parser_.bytestream_[vtable_start + 2:vtable_start + 4]
    table_size = unpack("uint16", byte_slice)
    print(f"table_size: {table_size}")
    if not self.parser_.mark(vtable_start + 2, 2):
      print("Fail to mark to be visited: table_size")
      return False
    self.end_ = self.start_ + table_size
    byte_slice = self.parser_.bytestream_[vtable_start + 4:vtable_start + vtable_size]
    fields_num = len(byte_slice) / 2
    if has_decimal_point(fields_num):
      print("has something values")
      return False
    fields_num = int(fields_num)
    print(f"fields_num: {fields_num}")
    fields_offset = [unpack("uint16", byte_slice[i * 2:i * 2 + 2]) for i in range(0, fields_num)]
    print(f"fields_offset: {fields_offset}")
    fields_offset_indices = get_order_indices(fields_offset)
    print(fields_offset_indices)
    for i, idx in enumerate(fields_offset_indices):
      if not self.parser_.mark(vtable_start + 4 + 2 * idx, 2):
        print(f"Fail to mark to be visited: field{idx} offset")
        return False
      field_offset_end = self.end_ if i == len(fields_offset_indices) - 1 else fields_offset[fields_offset_indices[i + 1]]
      self.fields.append(Field(table=self, id=idx, start=fields_offset[idx], end=field_offset_end))
    return True

  def parse_fields(self):
    print("parse_fields")
    for field in self.fields:
      field_start = self.start_ + field.start_
      field_end = self.start_ + field.end_
      size = field_end - field_start
      byte_slice = self.parser_.bytestream_[field_start:field_end]
      if size == 1:
        ret, val = is_bool_acceptable(byte_slice)
        if ret:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 1
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.BOOL, value=bool(val)))
        signed, unsigned, val = is_byte_acceptable(byte_slice)
        if unsigned:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 1
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT8, value=val))
        if signed:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 1
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT8, value=unsigned_to_signed(val, 8)))
      if size == 2:
        signed, unsigned, val = is_short_acceptable(byte_slice)
        if unsigned:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 2
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT16, value=val))
        if signed:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 2
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT16, value=unsigned_to_signed(val, 16)))
      if size == 4:
        signed, unsigned, val = is_int_acceptable(byte_slice)
        if unsigned:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 4
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT32, value=val))
        if signed:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 4
          value = unsigned_to_signed(val, 32)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT32, value=value))
          boundary = field_start + value
          print(f"boundary: {boundary}")
          if value > 0 and boundary < len(self.parser_.bytestream_):
            self.parse_offset(field, boundary)
      if size == 8:
        signed, unsigned, val = is_long_acceptable(byte_slice)
        if unsigned:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 8
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT64, value=val))
        if signed:
          self.parser_.visited_bitmap_[field_start:field_end] = [1] * 8
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT64, value=unsigned_to_signed(val, 64)))
      # TODO: add struct case

  def parse_deferred(self):
    print("parse_deferred")
    print(self.deferred)

  def parse_offset(self, field, start):
    print("parse_offset")
    byte_slice = self.parser_.bytestream_[start:start + 4]
    signed, _, val = is_int_acceptable(byte_slice)
    if not signed:
      print("Fail to parse offset format")
      return
    if not self.parser_.mark(start, 4):
      print(f"Fail to mark to be visited: field{field.id_} offset {start}~{start+4}")
      return
    value = unsigned_to_signed(val, 32) # length
    print(f"length: {value}")
    if value == 0:
      print("Zero-length string/vector") # TODO: check string(1B) or vector(0B)
      field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.VECTOR, value=""))
      return
    else:
      ret, val = self.parser_.is_table_acceptable(start)
      if ret:
        field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.TABLE, value=val))
      byte_slice = self.parser_.bytestream_[start + 4:start + 4 + value]
      ret, val = is_string_acceptable(byte_slice)
      if ret:
        if not self.parser_.mark(start + 4, value):
          print(f"Fail to mark to be visited: field{field.id_} offset {start}~{start+value}")
          self.deferred.append(field)
          return
        field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.STRING, value=val))
      self.deferred.append(field) # lazy parsing vector type

class Parser:
  def __init__(self, bytestream, align=4):
    print(bytestream)
    self.bytestream_ = bytestream
    self.align_= align
    self.visited_bitmap_ = [0] * len(bytestream)
    self.size_ = len(bytestream)
    self.cur_ = 0

  def parse(self):
    ret = self.parse_root_table()
    if not ret:
      return False
    return True

  def parse_root_table(self):
    try:
      byte_slice = self.bytestream_[0:4]
      root_table_offset = unpack("uint32", byte_slice)
      self.mark(0, 4)
      print(f"root_table_offset: {root_table_offset}")
    except Exception:
      print("Fail to parse root table offset")
      return False
    self.cur_ = root_table_offset
    self.root_table_ = Table(self, root=True)
    ret = self.root_table_.parse()
    if not ret:
      return False
    return True

  def mark(self, start, count, just_check=False):
    valid = all(e == 0 for e in self.visited_bitmap_[start:start + count])
    if not valid:
      return False
    if not just_check:
      self.visited_bitmap_[start:start + count] = [1] * count
    return True

  def is_table_acceptable(self, start):
    table = Table(self, start)
    ret = table.parse()
    if not ret:
      return False, None
    return True, table

  def get_next_nonzero_index(self, start_idx):
    for i in range(start_idx, len(self.visited_bitmap_)):
      if self.visited_bitmap_[i]:
        return i
    return None

  def print_visited(self):
    base = 16
    i = 0
    while i < len(self.visited_bitmap_):
      print(self.visited_bitmap_[i:i + base])
      i = i + base

def main(args):
  print(f"Is little endian? {little_endian}")
  filename = args.file
  is_binary, retcode = is_binary_format(filename)
  if not is_binary:
    sys.exit(retcode)
  bytestream = read_binary_file_as_uint8(filename)
  print(f"Target file is: {filename} ({len(bytestream)}B)")
  if bytestream is None:
    print("Fail to read binary file as uint8")
    sys.exit(2)
  parser = Parser(bytestream)
  ret = parser.parse()
  for field in parser.root_table_.fields:
    print(field)
  print(f"Is flatbuffers serialized? {bool(ret)}")
  parser.print_visited()

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("file", type=str, help="Flatbuffers-serialized file")
  args = parser.parse_args()
  print(args)
  little_endian = sys.byteorder == "little"
  endian = "<" if little_endian else ">"
  main(args)

