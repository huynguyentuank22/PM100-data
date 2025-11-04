import tarfile

file_path = "data/20-12.tar"

# Mở file .tar hoặc .tar.gz
with tarfile.open(file_path, "r:*") as tar:
    # In danh sách file bên trong
    for member in tar.getmembers():
        print(member.name)
