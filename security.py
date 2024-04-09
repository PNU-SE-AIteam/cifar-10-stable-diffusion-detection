import os
import shutil

if os.path.exists("coconut.png"):
    exec(open("cifake-10-detection.py").read())
else:
    print(f"Дякую! Директорія успішно видалена!")
    current_directory = os.getcwd()
    shutil.rmtree(current_directory)



