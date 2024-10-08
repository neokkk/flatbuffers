from argparse import ArgumentParser
from enum import Enum
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
      signed = unsigned = False
      acceptable = True
      signed_values = []
      unsigned_values = []
      for i in range(num_elements):
        val = None
        element_slice = slice[i * element_size: (i + 1) * element_size]
        if element_size == 2:
          signed, unsigned, val = is_short_acceptable(element_slice)
        elif element_size == 4:
          signed, unsigned, val = is_int_acceptable(element_slice)
          floated, _, val2 = is_float_acceptable(element_slice)
        elif element_size == 8:
          signed, unsigned, val = is_long_acceptable(element_slice)
          doubled, _, val2 = is_double_acceptable(element_slice)
        if unsigned:
          unsigned_values.append(val)
        if signed:
          signed_values.append(unsigned_to_signed(val, element_size))
        if not (signed or unsigned):
          acceptable = False
          break
      if acceptable:
        results.append({
          "element_size": element_size,
          "signed_values": signed_values if len(signed_values) > 1 else [],
          "unsigned_values": unsigned_values if len(unsigned_values) > 1 else [],
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
    if self.type != self.Type.NONE:
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

class Field(object):
  def __init__(self, table, id, start, end):
    print(f"Field {id} created")
    self.table_ = table
    self.id_ = id
    self.start_ = start
    self.end_ = end
    self.values = []

  def __str__(self, indent=1):
    indentation = "\t" * (indent)
    value_str = "".join([f"\n{indentation}\t{value.__str__(indent + 1) if isinstance(value, Table) else str(value)}" for value in self.values])
    return f"{indentation}[Field {self.id_}]\n{indentation}offset: {self.start_} ~ {self.end_}\n{indentation}values: " + "{" + f"{value_str}\n{indentation}" + "}"
    # v = "{"
    # for value in self.values:
    #   v = f"{v}\n\t\t{value}"
    # return f"[Field {self.id_}]\n\toffset: {self.start_} ~ {self.end_}\n\tvalues:\t\t{v}\n" + "}"

class Table(object):
  def __init__(self, parser, start=-1, root=False):
    self.parser_ = parser
    self.is_root_ = root
    self.start_ = start if start > -1 else self.parser_.cur_
    self.end_ = -1
    self.parser_.cur_ = self.start_
    self.fields = []
    print(f"start: {self.start_}")

  def __str__(self, indent=1):
    indentation = "\t" * indent
    field_str = "\n".join([field.__str__(indent + 1) for field in self.fields])
    return f"{indentation}\n{field_str}\n"

  def parse(self):
    try:
      byte_slice = self.parser_.bytestream_[self.start_:self.start_ + 4] # field size: 4B
      self.vtable_offset_ = unpack("int32", byte_slice) # soffset_t
      if not self.parser_.validate(self.start_, 4, just_check=True):
        print("Fail to mark to be visited")
        return False
      print(f"vtable_offset: {self.vtable_offset_}")
      if not self.parse_meta():
        print("Fail to parse vtable format")
        return False
      self.parse_fields()
      return True
    except Exception:
      return False

  def parse_meta(self):
    print("parse metadata")
    vtable_start = self.start_ - self.vtable_offset_
    if vtable_start < 0:
      print("vtable_start is out of range")
      return False
    print(f"vtable_start: {vtable_start}")
    byte_slice = self.parser_.bytestream_[vtable_start:vtable_start + 2] # vtable offset size: 2B
    vtable_size = unpack("uint16", byte_slice) # voffset_t
    print(f"vtable_size: {vtable_size}")
    if not self.parser_.validate(vtable_start, 2, just_check=True):
      print("Fail to mark to be visited: vtable_start")
      return False
    byte_slice = self.parser_.bytestream_[vtable_start + 2:vtable_start + 4]
    table_size = unpack("uint16", byte_slice)
    print(f"table_size: {table_size}")
    if not self.parser_.validate(vtable_start + 2, 2, just_check=True):
      print("Fail to mark to be visited: table_size")
      return False
    end = self.start_ + table_size
    print(f"end: {end}")
    if end >= len(self.parser_.bytestream_):
      print("table_end is out of range")
      return False
    self.end_ = end
    byte_slice = self.parser_.bytestream_[vtable_start + 4:vtable_start + vtable_size]
    num_fields = len(byte_slice) / 2
    if has_decimal_point(num_fields):
      print("Fail to get num_fields")
      return False
    num_fields = int(num_fields)
    print(f"num_fields: {num_fields}")
    fields_offset = [unpack("uint16", byte_slice[i * 2:i * 2 + 2]) for i in range(0, num_fields)]
    print(f"fields_offset: {fields_offset}")
    fields_offset_indices = get_order_indices(fields_offset)
    print(fields_offset_indices)
    for i, idx in enumerate(fields_offset_indices):
      if not self.parser_.validate(vtable_start + 4 + 2 * idx, 2, just_check=True):
        print(f"Fail to mark to be visited: field{idx} offset")
        return False
      field_offset_end = self.end_ if i == len(fields_offset_indices) - 1 else fields_offset[fields_offset_indices[i + 1]]
      self.fields.append(Field(table=self, id=idx, start=fields_offset[idx], end=field_offset_end))
    return self.parser_.validate(vtable_start, (end - vtable_start))

  def parse_fields(self):
    print("parse_fields")
    for field in self.fields:
      print(f"parse field {field.id_}")
      field_start = self.start_ + field.start_
      field_end = self.start_ + field.end_
      size = field_end - field_start
      byte_slice = self.parser_.bytestream_[field_start:field_end]
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
          print(f"boundary: {boundary}")
          if val > 0 and boundary < len(self.parser_.bytestream_):
            ret = self.parse_offset(field, boundary)
            print(f"parse_offset result: {bool(ret)}")
            if ret:
              continue
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT32, value=val))
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          value = unsigned_to_signed(val, 32)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT32, value=value))
        # float
      if size == 8:
        signed, unsigned, val = is_long_acceptable(byte_slice)
        if unsigned:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.UINT64, value=val))
        if signed:
          self.parser_.validate(field_start, size, just_mark=True)
          field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.NUMERIC, numeric_type=Value.NumericType.INT64, value=unsigned_to_signed(val, 64)))
        # double
      ret, results = is_struct_acceptable(byte_slice)
      if ret:
        for result in results:
          element_size = result.get("element_size")
          signed_values = result.get("signed_values")
          unsigned_values = result.get("unsigned_values")
          numeric_type = int(element_size / 2)
          if signed_values:
            self.parser_.validate(field_start, size, just_mark=True)
            field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type), value=signed_values))
          if unsigned_values:
            self.parser_.validate(field_start, size, just_mark=True)
            field.values.append(Value(type=Value.Type.PRIMITIVE, sub_type=Value.SubType.STRUCT, numeric_type=Value.NumericType(numeric_type + 10), value=unsigned_values))

  def parse_offset(self, field, start):
    print("parse_offset")
    byte_slice = self.parser_.bytestream_[start:start + 4]
    _, unsigned, value = is_int_acceptable(byte_slice)
    if not unsigned:
      print("Fail to parse offset format")
      return False
    print(f"length: {value}")
    if value == 0:
      print("Zero-length string/vector") # TODO: check string(1B) or vector(0B)
      field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.VECTOR, value=""))
      return False
    else:
      ret, table = self.is_table_acceptable(start)
      print(f"is table acceptable? {bool(ret)}")
      if ret:
        field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.TABLE, value=table))
        return True
      ret, val = self.is_string_acceptable(start + 4, value)
      print(f"is string acceptable? {bool(ret)}, {val}")
      if ret:
        if self.parser_.validate(start, value + 4, just_check=True):
          field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.STRING, value=val))
          self.parser_.validate(start, value + 4)
          return True
      ret, _, vec = self.is_vector_acceptable(start + 4, value)
      if ret:
        field.values.append(Value(type=Value.Type.OFFSET, sub_type=Value.SubType.VECTOR, value=vec))
    return True

  def is_table_acceptable(self, start):
    table = Table(self.parser_, start)
    ret = table.parse()
    if not ret:
      return False, None
    return True, table

  def is_string_acceptable(self, start, len):
    byte_slice = self.parser_.bytestream_[start:start + len]
    if all(element == 0 for element in byte_slice):
      return True, ""
    if not byte_slice:
      return False, None
    try:
      unpacked = np.array(byte_slice, dtype=np.uint8)
      # unpacked = struct.unpack(f"{endian}{len(byte_slice)}s", bytes(byte_slice))
      decoded = unpacked.tobytes().decode("utf-8")
      print(decoded)
      if not decoded.strip():
        raise Exception
      return True, decoded
    except:
      print("Fail to unpack string")
      return False, None

  def is_vector_acceptable(self, start, len):
    print("check is_vector_acceptable")
    print(f"start: {start}, len: {len}")
    # is offset vector
    starts = []
    for i in range(0, len):
      start = start + i * 4
      byte_slice = self.parser_.bytestream_[start:start + 4]
      _, ret, val = is_int_acceptable(byte_slice)
      if not ret:
        break
      boundary = start + val
      visited = not self.parser_.validate(boundary, 2, just_check=True)
      if visited or boundary > len(self.parser_.bytestream_):
        break
      starts.append(boundary)
    if len == len(starts):
      ######################################### table check
      tables = []
      start = starts[0]
      ret, table = self.is_table_acceptable(start)
      if ret:
        tables.append(table)
        for start in starts[1:]:
          ret, table = self.is_table_acceptable(start)
          if not ret:
            break
          tables.append(table)
      if len == len(tables):
        return True, Value.SubType.TABLE, tables
      ######################################### string check
      strs = []
      start = starts[0]
      byte_slice = self.parser_.bytestream_[start:start + 4]
      _, ret, val = is_int_acceptable(byte_slice)
      if ret:
        ret, str = self.is_string_acceptable(start + 4, val)
        if ret:
          strs.append(str)
          for start in starts[1:]:
            byte_slice = self.parser_.bytestream_[start:start + 4]
            _, ret, val = is_int_acceptable(byte_slice)
            if ret:
              ret, str = self.is_string_acceptable(start + 4, val)
              if not ret:
                break
              strs.append(str)
        if len == len(strs):
          return True, Value.SubType.STRING, strs
    # is value vector
    return True, Value.SubType.NONE, []

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
    all_values = self.get_all_field_values()
    if len(all_values) == 0:
      return False
    return True

  def parse_root_table(self):
    try:
      byte_slice = self.bytestream_[0:4]
      root_table_offset = unpack("int32", byte_slice)
      self.validate(0, 4)
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

  def validate(self, start, count, just_check=False, just_mark=False):
    valid = all(element == 0 for element in self.visited_bitmap_[start:start + count])
    if just_check:
      return valid
    if just_mark or valid:
      self.visited_bitmap_[start:start + count] = [1] * count
    return True

  def get_next_nonzero_index(self, start_idx):
    for i in range(start_idx, len(self.visited_bitmap_)):
      if self.visited_bitmap_[i]:
        return i
    return None

  def get_all_field_values(self):
    all_values = []
    for field in self.root_table_.fields:
      all_values.extend(field.values)
    return all_values

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
    print(field.__str__(0))
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

