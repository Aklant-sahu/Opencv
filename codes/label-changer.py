import os
import numpy as np
path="codes/ftc-data/Dataset"

files=os.listdir(path)

for item in files:
    z=item.split(".")
    
    if z[-1]=='txt':
        print(z)
        y=item.split("-")
        print(y)
        p=path+"/"+item
        f = open(p, "r")
        x=f.read()
        x=x.split(" ")
        print(y[0])
        print(x[0])
        
        if y[0]=='square':
            x[0]='0'
            
        
        if y[0]=='duck':
            x[0]='1'
        if y[0]=='circle':
            x[0]='2'
        f.close()
        f = open(p, "w")
        print(x[0])
        s=" "
        # print(s.join(x))
        f.write(s.join(x))
        f.close()

print("done")