import os

bin_path = "/media/mariandawud/int_ssd/data/mS51_49.dat"  # <- change this
size_bytes = os.path.getsize(bin_path)
print("File size (bytes):", size_bytes)

print("-----------Number of channels check-----------")
for nchan in (62, 63, 126, 128, 135):
    nsamp, rem = divmod(size_bytes, 2 * nchan)  # 2 bytes per int16
    print(f"nchan={nchan}: nsamp={nsamp}, remainder={rem}")
print("-----------Recording duration check-----------")
print(size_bytes / (135 * 2 * 20000))
