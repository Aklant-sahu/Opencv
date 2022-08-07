import PIL
from PIL import Image
inputimage=Image.open('D:\opencv\hair-club.png')
pixelmap=inputimage.load()
width,height=inputimage.size
for i in range(width):
    for j  in range(height):
        r,g,b,p=inputimage.getpixel((i,j))
        if ((g-r)>5 ) and ((g-b)>5):
            r=r+10
            b=b-20
            g=(r+b)//2
        elif (((g-r)>8) and ((b-r)>8)) and (b<200) and(r<100) and(g<95):
            b=b/2
            g=g/2
            r=r/2


        pixelmap[i,j]=(int(r),int(g),int(b))

inputimage.show()
inputimage.save('D:\opencv\output-2.png')

