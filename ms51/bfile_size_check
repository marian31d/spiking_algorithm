import os

bin_path = "/media/mariandawud/int_ssd/data/mS51_49.dat"  # <- change this
size_bytes = os.path.getsize(bin_path)
print("File size (bytes):", size_bytes)

for nchan in (126, 128):
    nsamp, rem = divmod(size_bytes, 2 * nchan)  # 2 bytes per int16
    print(f"nchan={nchan}: nsamp={nsamp}, remainder={rem}")
