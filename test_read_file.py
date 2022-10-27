from py_script.utils import read_file

if __name__ == "__main__":
    mpc = read_file("./test/data/epri21.m")
    print(mpc)
