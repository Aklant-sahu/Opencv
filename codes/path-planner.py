import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
print([i for i in dir(cv2) if 'EVENT' in i])
scalingFactor=2
global px
global py
px=[]
py=[]

global poi
poi=np.array([])

def area2(a,b,c) :
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
def dist(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
def curvature(x,y,length):
    curve=[]
    for item in range(length):
        if(item==0):
            curve.append(0)
        elif(item==length-1):
            curve.append(0)
        else:
            prev_point=[x[item-1],y[item-1]]
            curr_point=[x[item],y[item]]
            next_point=[x[item+1],y[item+1]]
            areaoftriangle=area2(prev_point,curr_point,next_point)
            areaoftriangle=abs(areaoftriangle/2)
            l1=dist(prev_point,curr_point)
            l2=dist(prev_point,next_point)
            l3=dist(curr_point,next_point)
            curvature=(4*areaoftriangle)/(l1*l2*l3)
            curve.append(curvature)
    return curve

def trapezoidalProfile(maxvelocity,maxaccelaration,startvelocity,endvelocity,starttime,endtime,points,totalDist):
    ttmaxvel=maxvelocity/maxaccelaration
    disttravelinaccelaration=(maxaccelaration*(ttmaxvel**2))/2
    disttravelindeaccelaration=disttravelinaccelaration
    disttravelwithconstantvelocity=totalDist-2*disttravelinaccelaration
    ttconstvel=disttravelwithconstantvelocity/maxvelocity
    tttaken=ttconstvel +2*ttmaxvel
    if tttaken>endtime-starttime:
        print("Changing time interval to reach target within max velocity")
        endtime=endtime-starttime+tttaken
        print(f"New Time interval {endtime-starttime} s ")
    else:
        endtime=starttime+tttaken
        print("Changing time interval to reach target within max velocity")
        print(f"New Time interval {endtime-starttime} s ")
    print(f"Time taken to reach max velocity {ttmaxvel} s ")
    print(f"Time under constant velocity {ttconstvel} s ")
    

    point=points//3
    
    conststopTime=starttime+ttmaxvel+ttconstvel
    velocity=[]
    dist=[]
    timing=[]
    accelaration=[]
    accel=np.linspace(starttime,ttmaxvel,point,endpoint=False)
    
    constaccel=np.linspace(ttmaxvel,starttime+ttmaxvel+ttconstvel,point,endpoint=False)
    deccel=np.linspace(starttime+ttmaxvel+ttconstvel,endtime,point)
    for item in accel:
        velocity.append(startvelocity+(maxaccelaration*(item-starttime)))
        timing.append(item)
        accelaration.append(maxaccelaration)
    for item in constaccel:
        velocity.append(maxvelocity)
        timing.append(item)
        accelaration.append(0)
    for item in deccel:
        velocity.append(maxvelocity-(maxaccelaration*(item-conststopTime)))
        timing.append(item)
        accelaration.append(-maxaccelaration)
    distCovered=2*(disttravelinaccelaration)+(disttravelwithconstantvelocity)
    print(f"Dist Covered {distCovered} ")
    print(f"Max velocity {maxvelocity} ")
    print(f"Start time {starttime} ")
    print(f"End time {endtime} ")

    return velocity,timing,accelaration
# find the a & b points
def distances(x,y):
    dist=[]
    for i in range(len(x)-1):
        dist.append(((x[i]-x[i+1])**2 + (y[i]-y[i+1])**2)**0.5)
    return sum(dist)
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])
def callback1(event,x,y,flags,param):  # This is the specific format with the specific parameters it take in order to create a callback
    if event==cv2.EVENT_LBUTTONDOWN:
        font=cv2.FONT_ITALIC
        text =f'X:{x} Y:{y}'
        p.append([x,y])
        

        cv2.putText(img,text,(x,y),font,0.5,(0,255,0),2) # since it is a callback function so i cant reassign this to img and show 
        # as there is noyhing as img as we are not sending the imaage arg
        cv2.imshow("image",img)
    if event==cv2.EVENT_RBUTTONDOWN:
        # font=cv2.FONT_ITALIC
        if(len(p)>1):
            # plt.close()
            points = np.array(p)

            path = evaluate_bezier(points,no_control_points)
            # x1, y1 = points[:,0], points[:,1]
            px, py = path[:,0], path[:,1]
            dist=distances(px,py)/scalingFactor
            print(dist)
            # print(px)
            poi=np.array(list(zip(px,py)),dtype=int)
            # print(len(poi))
            velocity,times,accelaration=trapezoidalProfile(39.37,39.37,0,0,0,20,len(poi),dist) 
            curve=curvature(px,py,len(px))
            # print(len(velocity))
            Trajectory=[]
            
            for i in range(len(poi)):
                curr={'x':"",'y':"",'velocity':"","accelaration":"","timings":"","curvature":""}
                

                curr['x']=poi[i][0]
                curr['y']=poi[i][1]
                curr['velocity']=velocity[i]
                curr['accelaration']=accelaration[i]
                curr['timings']=times[i]
                curr['curvature']=curve[i]
                Trajectory.append(curr)
            # print(Trajectory)
            print(curve)
            for num in curve:
                if num < 0:
                    print(num, end = " ")
            # json_obj=json.dumps(curr)
            
            # with open('data.json', 'w', encoding='utf-8') as f:
            #     json.dump(curr, f, ensure_ascii=False, indent=4)

   
            # plt.plot(px, py, 'b-')
            # plt.show()
            
            for i in range(len(px)-1):
                cv2.line(img,tuple(poi[i]),tuple(poi[i+1]),color=(255,255,255),thickness=2)
            
        # text =f'[{img[y,x,0]},{img[y,x,1]},{img[y,x,2]}]'
        # cv2.putText(img,text,(x,y),font,0.5,[255,255,255],2) # since it is a callback function so i cant reassign this to img and show 
        # as there is noyhing as img as we are not sending the imaage arg
        
        cv2.imshow("image",img)

def show(scalingFactor=2):
    global img
    global k
    global p
    global no_control_points
    p=[]
    no_control_points=60
    img=cv2.imread("C:/Users/asus/Downloads/2022-field.png",1)
    img=cv2.resize(img,(704*scalingFactor,360*scalingFactor))
    cv2.imshow("image",img)
    cv2.setMouseCallback("image",callback1)
    k=cv2.waitKey(0)
    if k==ord('q'):
        cv2.destroyAllWindows()
    if k==ord('r'):
        # print(poi)
        # print(py)
        cv2.destroyAllWindows()
        show()
    # if k==ord('s'):
    #     print(px)
    #     print(py)
        


show()
   
    