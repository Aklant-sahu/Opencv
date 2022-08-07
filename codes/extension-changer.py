import os,sys
folder="codes/ftc-data/Dataset/labels/"
for filename in os.listdir(folder):
    infilename = os.path.join(folder,filename)
    if not os.path.isfile(infilename): continue
    oldbase = os.path.splitext(filename)
    print(infilename)

    newname = infilename.replace('.xml.txt', '.txt')
    print(newname)
    output = os.rename(infilename, newname)
