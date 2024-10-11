#!/usr/bin/env python3
#
# Not supported to deserialization
# - Nested vector (vector - vector)
# - Differentiated struct (tuple values)
#

from argparse import ArgumentParser
from collections.abc import Sequence
from enum import Enum
import math
import numpy as np
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
  "float": "f",
  "double": "d",
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

def unpack(type, slice):
  format_char = type_to_struct_format.get(type, None)
  if not format_char:
    raise Exception(f"Fail to unpack {type} type as struct format")
  return struct.unpack(f"{endian}{format_char}", bytes(slice))[0]

def has_decimal_point(num):
  return num % 1 != 0

def get_order_indices(list, order="asc"):
  reverse = order == "desc"
  return sorted(range(len(list)), key=lambda x: list[x], reverse=reverse)

def signed_to_unsigned(value, e):
  base = 2 ** e
  return (value + base) % base

def unsigned_to_signed(value, e):
  base = 2 ** (e - 1)
  if value >= base:
    return value - 2 ** e
  return value

def is_array_like(obj):
  return isinstance(obj, (Sequence, np.ndarray)) and not isinstance(obj, str)

def is_bool_acceptable(slice):
  _, unsigned, val = is_byte_acceptable(slice)
  if unsigned and (val == 0 or val == 1):
    return True, val
  return False, None

def is_value_acceptable(slice, type, list, just_signed=False):
  unsigned_type = f"u{type}"
  signed = unsigned = False
  val = -1
  try:
    val = unpack(type, slice)
    if len(slice) != list:
      raise Exception
    signed = True
  except Exception:
    print(f"{type} unacceptable")
  if not just_signed:
    try:
      val = unpack(unsigned_type, slice)
      unsigned = True
    except Exception:
      print(f"{unsigned_type} unacceptable")
  return signed, unsigned, val

def is_byte_acceptable(slice):
  return is_value_acceptable(slice, "int8", 1)

def is_short_acceptable(slice):
  return is_value_acceptable(slice, "int16", 2)

def is_int_acceptable(slice):
  return is_value_acceptable(slice, "int32", 4)

def is_long_acceptable(slice):
  return is_value_acceptable(slice, "int64", 8)

def is_float_acceptable(slice):
  return is_value_acceptable(slice, "float", 4, just_signed=True)

def is_double_acceptable(slice):
  return is_value_acceptable(slice, "double", 8, just_signed=True)

def is_struct_acceptable(slice):
  size = len(slice)
  possible_sizes = [1, 2, 4, 8]
  results = []
  for element_size in possible_sizes:
    if size % element_size == 0:
      num_elements = size // element_size
      signed = unsigned = real = False
      acceptable = True
      signed_values = []
      unsigned_values = []
      real_values = []
      for i in range(num_elements):
        val = val2 = None
        element_slice = slice[i * element_size: (i + 1) * element_size]
        if element_size == 2:
          signed, unsigned, val = is_short_acceptable(element_slice)
        elif element_size == 4:
          signed, unsigned, val = is_int_acceptable(element_slice)
          real, _, val2 = is_float_acceptable(element_slice)
        elif element_size == 8:
          signed, unsigned, val = is_long_acceptable(element_slice)
          real, _, val2 = is_double_acceptable(element_slice)
        if unsigned:
          unsigned_values.append(val)
        if signed:
          signed_values.append(unsigned_to_signed(val, element_size))
        if real:
          real_values.append(val2)
        if not (signed or unsigned or real):
          acceptable = False
          break
      if acceptable:
        results.append({
          "element_size": element_size,
          "signed_values": signed_values if len(signed_values) > 1 else [],
          "unsigned_values": unsigned_values if len(unsigned_values) > 1 else [],
          "real_values": real_values if len(real_values) > 1 else [],
        })
  if not results:
    return False, []
  return True, results

class Value(object):
  def __init__(self, type=None, sub_type=None, numeric_type=None, value=None):
    self.type = self.Type.NONE if type is None else type
    self.sub_type = self.SubType.NONE if sub_type is None else sub_type
    self.numeric_type = self.NumericType.NONE if numeric_type is None else numeric_type
    self.value = value = value

  def __str__(self):
    v = ""
    if self.type == self.Type.NONE:
      v = f"{self.type}"
    if self.sub_type != self.SubType.NONE:
      v = f"{v}, {self.sub_type}" if v else f"{self.sub_type}"
    if self.numeric_type != self.NumericType.NONE:
      v = f"{v}, {self.numeric_type}" if v else f"{self.numeric_type}"
    return f"{v}: {self.value}"

  class Type(Enum):
    NONE = -1
    UNKNOWN = 0
    PRIMITIVE = 1 # bool, enum, numeric, struct
    OFFSET = 2 # string, vector, table

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
    FLOAT = 23
    DOUBLE  = 24

class Defer(object): # OFFSET, VECTOR, <NUMERIC_TYPE>
  def __init__(self, start, field, competitor=""):
    self.start = start
    self.field = field
    self.competitor = competitor

class Field(object):
  def __init__(self, table, id, start, end):
    self.table_ = table
    self.id_ = id
    self.start_ = start
    self.end_ = end
    self.values = []

  def __str__(self, indent=1):
    indentation = "\t" * (indent)
    value_str = "".join([f"\n{indentation}\t{value.__str__(indent + 1) if isinstance(value, Table) else str(value)}" for value in self.values])
    return f"{indentation}[Field {self.id_}]\n{indentation}offset: {self.start_} ~ {self.end_}\n{indentation}values: " + "{" + f"{value_str}\n{indentation}" + "}"

class Table(object):
  def __init__(self, parser, start=0, root=False):
    self.parser_ = parser
    self.is_root_ = root
    self.start_ = start
    self.end_ = -1
    self.vtable_start_ = -1
    self.vtable_size_ = -1
    self.table_size_ = -1
    self.fields_ = []

  def __str__(self, indent=1):
    indentation = "\t" * indent
    field_str = "\n".join([field.__str__(indent + 1) for field in self.fields_])
    return f"{indentation}\n{field_str}"

  def parse(self):
    try:
      byte_slice = self.parser_.bytestream_[self.start_:self.start_ + 4] # field size: 4B
      self.vtable_offset_ = unpack("int32", byte_slice) # soffset_t
      if not self.parser_.validate(self.start_, 4, just_check=True):
        return False
      if not self.parse_meta():
        return False
      self.parse_fields()
      return True
    except Exception:
      return False

  def parse_meta(self):
    vtable_start = self.start_ - self.vtable_offset_
    if vtable_start < 0:
      return False
    self.vtable_start_ = vtable_start
    byte_slice = self.parser_.bytestream_[vtable_start:vtable_start + 2] # vtable offset size: 2B
    vtable_size = unpack("uint16", byte_slice) # voffset_t
    if vtable_size < 1:
      return False
    self.vtable_size_ = vtable_size
    if not self.parser_.validate(vtable_start, 2, just_check=True):
      return False
    byte_slice = self.parser_.bytestream_[vtable_start + 2:vtable_start + 4]
    table_size = unpack("uint16", byte_slice)
    if table_size < 1:
      return False
    self.table_size_ = table_size
    if not self.parser_.validate(vtable_start + 2, 2, just_check=True):
      return False
    end = self.start_ + table_size
    if end >= self.parser_.size_:
      return False
    self.end_ = end
    byte_slice = self.parser_.bytestream_[vtable_start + 4:vtable_start + vtable_size]
    num_fields = len(byte_slice) / 2
    if has_decimal_point(num_fields):
      return False
    num_fields = int(num_fields)
    fields_offset = [unpack("uint16", byte_slice[i * 2:i * 2 + 2]) for i in range(0, num_fields)]
    fields_offset_indices = get_order_indices(fields_offset)
    for i, idx in enumerate(fields_offset_indices):
      if not self.parser_.validate(vtable_start + 4 + 2 * idx, 2, just_check=True):
        return False
      field_offset_end = table_size if i == len(fields_offset_indices) - 1 else fields_offset[fields_offset_indices[i + 1]]
      field = Field(table=self, id=idx, start=fields_offset[idx], end=field_offset_end)
      self.fields_.append(field)
    return self.parser_.validate(vtable_start, (end - vtable_start))

  def parse_fields(self):
    for field in self.fields_:
      field_start = self.start_ + field.start_
      field_end = self.start_ + field.end_
      size = field_end - field_start
      byte_slice = self.parser_.bytestream_[field_start:field_end]
      ######################################## numeric check
      if size == 1:
        ret, val = is_bool_acceptable(byte_slice)
        if ret:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.BOOL, value=bool(val)))
        signed, unsigned, val = is_byte_acceptable(byte_slice)
        if unsigned:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT8, value=val))
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT8, value=unsigned_to_signed(val, 8)))
      if size == 2:
        signed, unsigned, val = is_short_acceptable(byte_slice)
        if unsigned:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT16, value=val))
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT16, value=unsigned_to_signed(val, 16)))
      if size == 4:
        signed, unsigned, val = is_int_acceptable(byte_slice)
        if unsigned:
          boundary = field_start + val
          if val > 0 and boundary < self.parser_.size_:
            ret = self.parse_offset(field, boundary)
            if ret:
              continue
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT32, value=val))
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          value = unsigned_to_signed(val, 32)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT32, value=value))
        # float
        signed, _, val = is_float_acceptable(byte_slice)
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.FLOAT, value=val))
      if size == 8:
        signed, unsigned, val = is_long_acceptable(byte_slice)
        if unsigned:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT64, value=val))
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT64, value=unsigned_to_signed(val, 64)))
        # double
        signed, _, val = is_double_acceptable(byte_slice)
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.DOUBLE, value=val))
      ######################################## struct check
      ret, results = is_struct_acceptable(byte_slice)
      if ret:
        for result in results:
          element_size = result.get("element_size")
          signed_values = result.get("signed_values")
          unsigned_values = result.get("unsigned_values")
          real_values = result.get("real_values")
          numeric_type = int(math.log2(element_size)) + 1
          if signed_values:
            self.parser_.validate(field_start, size, just_mark=True)
            field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type), value=signed_values))
          if unsigned_values:
            self.parser_.validate(field_start, size, just_mark=True)
            field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type + 10), value=unsigned_values))
          if real_values:
            self.parser_.validate(field_start, size, just_mark=True)
            field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type + 20), value=real_values))

  def parse_offset(self, field, start):
    byte_slice = self.parser_.bytestream_[start:start + 4]
    _, unsigned, value = is_int_acceptable(byte_slice)
    if not unsigned:
      return False
    if value == 0: # TODO: check string or vector
      self.parser_.validate(start, 4)
      field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.VECTOR, value=[]))
      return False
    else:
      ret, table = self.parser_.is_table_acceptable(start)
      if ret:
        field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.TABLE, value=table))
        return True
      str_ret, val = self.parser_.is_string_acceptable(start + 4, value)
      vec_ret, result = self.parser_.is_vector_acceptable(start + 4, value)
      vectors = result.get("vectors", [])
      sizes = result.get("sizes", [])
      t = result.get("type", Value.SubType.NONE)
      if not vectors and t is Value.SubType.UNKNOWN and not sizes:
        self.parser_.deferred.append(Defer(start=start, field=field, competitor=val if str_ret else ""))
        return True
      if vec_ret:
          for i, v in enumerate(vectors):
            s, l = sizes[i]
            ret = self.parser_.validate(s, l + 4 * value)
            field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.VECTOR, value=v))
          return True
      elif str_ret:
        ret = self.parser_.validate(start, value + 4)
        field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.STRING, value=val))
        return True
    return False

class Parser:
  def __init__(self, bytestream, align=4):
    self.bytestream_ = bytestream
    self.align_= align
    self.visited_bitmap_ = [0] * len(bytestream)
    self.deferred = [] # primitive value vector
    self.size_ = len(bytestream)

  def parse(self):
    ret = self.parse_root_table()
    if not ret:
      return False
    ret = self.parse_deferred() # parse primitive vector lazily to find end boundary
    all_values = self.get_all_field_values()
    if len(all_values) == 0:
      return False
    return True

  def parse_root_table(self):
    try:
      byte_slice = self.bytestream_[0:4]
      root_table_offset = unpack("int32", byte_slice)
      self.validate(0, 4)
    except Exception:
      print("Fail to parse root table offset")
      return False
    self.root_table_ = Table(self, start=root_table_offset, root=True)
    ret = self.root_table_.parse()
    if not ret:
      return False
    return True

  def parse_deferred(self):
    for defer in self.deferred:
      start = defer.start
      possible_end = self.get_next_nonzero_index(start)
      if possible_end < 0:
        print("Not exist nonzero index")
        continue
      comp = defer.competitor
      if comp:
        ret, val = self.is_string_acceptable(start + 4, possible_end - start - 4)
        if comp == val:
          self.validate(start, possible_end, just_mark=True)
          defer.field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.STRING, value=comp))
          continue
      pause = False
      if possible_end - start < 1:
        continue
      while not pause:
        byte_slice = self.bytestream_[start:possible_end]
        ret, results = is_struct_acceptable(byte_slice)
        if not ret:
          possible_end = possible_end - 1
          if self.visited_bitmap_[possible_end] == 1:
            pause = True
          continue
        size = possible_end - start
        for result in results:
          element_size = result.get("element_size", 0)
          signed_values = result.get("signed_values", [])
          unsigned_values = result.get("unsigned_values", [])
          real_values = result.get("real_values", [])
          numeric_type = int(math.log2(element_size)) + 1
          if signed_values:
            self.validate(start, size, just_mark=True)
            defer.field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type), value=signed_values))
          if unsigned_values:
            self.validate(start, size, just_mark=True)
            defer.field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type + 10), value=unsigned_values))
          if real_values:
            self.validate(start, size, just_mark=True)
            defer.field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type + 20), value=real_values))
        pause = True

  def is_table_acceptable(self, start):
    table = Table(self, start)
    ret = table.parse()
    if not ret:
      return False, None
    return True, table

  def is_string_acceptable(self, start, length):
    byte_slice = self.bytestream_[start:start + length]
    if all(element == 0 for element in byte_slice):
      return True, ""
    if not byte_slice:
      return False, None
    try:
      unpacked = np.array(byte_slice, dtype=np.uint8)
      decoded = unpacked.tobytes().decode("utf-8", errors="replace")
      stripped = decoded.rstrip("\x00")
      if not stripped:
        raise Exception
      return True, stripped
    except:
      print("Fail to unpack string")
      return False, None

  def is_vector_acceptable(self, start, length): # offset vector
    starts = []
    for i in range(0, length):
      s = start + i * 4
      byte_slice = self.bytestream_[s:s + 4]
      _, ret, val = is_int_acceptable(byte_slice)
      if not ret:
        break
      boundary = s + val
      if boundary > self.size_ or not self.validate(boundary, 2, just_check=True):
        break
      starts.append(boundary)
    if length == len(starts):
      ######################################## table check
      tables = []
      sizes = []
      s0 = starts[0]
      ret, table = self.is_table_acceptable(s0)
      if ret:
        vtable_start = getattr(table, "vtable_start_")
        vtable_size = getattr(table, "vtable_size_")
        table_size = getattr(table, "table_size_")
        tables.append(table)
        sizes.append((vtable_start, vtable_size + table_size))
        for st in starts[1:]:
          ret, tab = self.is_table_acceptable(st) # TODO: check field num same
          if not ret:
            break
          vtable_start = getattr(tab, "vtable_start_")
          vtable_size = getattr(tab, "vtable_size_")
          table_size = getattr(tab, "table_size_")
          tables.append(tab)
          sizes.append((vtable_start, vtable_size + table_size))
      if length == len(tables):
        result = {
          "vectors": tables,
          "sizes": sizes,
          "type": Value.SubType.TABLE,
        }
        return True, result
      ######################################## string check
      strs = []
      sizes = []
      s0 = starts[0]
      byte_slice = self.bytestream_[s0:s0 + 4]
      _, ret, val = is_int_acceptable(byte_slice)
      if ret:
        ret, str = self.is_string_acceptable(s0 + 4, val)
        if ret:
          strs.append(str)
          sizes.append((s0, val + 4))
          for st in starts[1:]:
            byte_slice = self.bytestream_[st:st + 4]
            _, ret, val = is_int_acceptable(byte_slice)
            if ret:
              ret, str = self.is_string_acceptable(st + 4, val)
              if not ret:
                break
              strs.append(str)
              sizes.append((st, val + 4))
        if length == len(strs): # all units are string
          result = {
            "vectors": strs,
            "sizes": sizes,
            "type": Value.SubType.STRING,
          }
          return True, result
    result = {
      "vectors": [],
      "sizes": [],
      "type": Value.SubType.UNKNOWN,
    }
    return True, result

  def validate(self, start, count, just_check=False, just_mark=False):
    valid = False if start + count >= self.size_ else True
    valid = all(element == 0 for element in self.visited_bitmap_[start:start + count]) if valid else False
    if just_check:
      return valid
    if just_mark or valid:
      self.visited_bitmap_[start:start + count] = [1] * count
    return True

  def get_next_nonzero_index(self, start):
    for i in range(start, len(self.visited_bitmap_)):
      if self.visited_bitmap_[i]:
        return i
    return self.size_ if self.visited_bitmap_[self.size_ - 1] == 0 else -1

  def get_all_field_values(self):
    all_values = []
    for field in self.root_table_.fields_:
      all_values.extend(field.values)
    return all_values

  def print_visited(self, start=0, end=0):
    base = 16
    i = start
    end = end if end and end < self.size_ else self.size_
    while i < end:
      e = i + base if i + base < end else end
      print(self.visited_bitmap_[i:e])
      i = i + base

def main(args):
  print(f"Is little endian? {little_endian}")
  filename = args.file
  is_binary, retcode = is_binary_format(filename)
  if not is_binary:
    sys.exit(retcode)
  bytestream = read_binary_file_as_uint8(filename)
  print(f"Target file is: {filename} ({len(bytestream)} Bytes)")
  if bytestream is None:
    print("Fail to read binary file as uint8")
    sys.exit(2)
  parser = Parser(bytestream)
  ret = parser.parse()
  print("=========================================")
  for field in parser.root_table_.fields_:
    print(field.__str__(0))
  print("=========================================")
  print(f"Is flatbuffers serialized? {bool(ret)}")
  if args.print:
    parser.print_visited()

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("file", type=str, help="Input flatbuffers-serialized file")
  parser.add_argument("-p", "--print", action="store_true", help="Print visited bitmap")
  args = parser.parse_args()
  little_endian = sys.byteorder == "little"
  endian = "<" if little_endian else ">"
  main(args)

