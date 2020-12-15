"""
Quickie little script to grab the body of data points we want to inspect
manually.
A. Knight
12/7
"""
from infuse_data import read_blogs, read_NYT

def get_data():
    blogs = read_blogs(verbose = True)
    nyt = read_NYT(verbose = True)
    ret = []
    while True:
        set = int(input("\nSelect dataset [1: blogs; 2: nyt]: "))
        idx = int(input("Index? "))
        data = (blogs if set == 1 else nyt)
        curr = data[idx]
        ret.append([set, idx, curr])
        con = input("Continue? [y/n]: ")
        if con == "n":
            return ret

def main():
    print("Grab as many document bodies as you'd like to inspect.")
    data = get_data()

    filename = input("Output file (include .txt): ")
    with open(filename, mode='a+') as myfile:
        for line in data:
            myfile.write("Taken from Blog Data\n" if line[0]==1 else "Taken from NYT Data\n")
            myfile.write("Item %d\n" % line[1])
            myfile.write("%s\n\n" % line[2])
    myfile.close()
    print("Data logged to local file %s" % filename)

main()
