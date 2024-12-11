import json
import struct
import sys

with open(sys.argv[1], 'rb') as f:
    length_of_header = struct.unpack('<Q', f.read(8))[0]
    header_data = f.read(length_of_header)
header = json.loads(header_data)
print(json.dumps(header, indent=4))
