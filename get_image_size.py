import os
import io
import struct

def get_image_metadata(file_path, max_size):
    size = os.path.getsize(file_path)
    if size > max_size:
        return size, -1, -1

    with io.open(file_path, "rb") as input:
        height = -1
        width = -1
        data = input.read(26)
        msg = " raised while trying to decode as JPEG."

        if (size >= 2) and data.startswith(b'\377\330'):
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF):
                        b = input.read(1)
                    while (ord(b) == 0xFF):
                        b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(
                            int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w) # type: ignore
                height = int(h) # type: ignore
            except Exception as e:
                raise Exception(e.__class__.__name__ + msg)
        else:
            raise Exception("Sorry, don't know how to get size for this file.")

        return size, width, height
