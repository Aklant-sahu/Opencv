import cv2
print([i for i in dir(cv2) if 'EVENT' in i])



import numpy as np
import matplotlib.pyplot as plt

# find the a & b points
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
            # print(px)
            poi=np.array(list(zip(px,py)),dtype=int)
            # plt.plot(px, py, 'b-')
            # plt.show()
            
            for i in range(len(px)-1):
                cv2.line(img,tuple(poi[i]),tuple(poi[i+1]),color=(255,255,255),thickness=2)
        # text =f'[{img[y,x,0]},{img[y,x,1]},{img[y,x,2]}]'
        # cv2.putText(img,text,(x,y),font,0.5,[255,255,255],2) # since it is a callback function so i cant reassign this to img and show 
        # as there is noyhing as img as we are not sending the imaage arg
        
        cv2.imshow("image",img)

def show():
    global img
    global k
    global p
    global no_control_points
    p=[]
    no_control_points=100
    img=cv2.imread("C:/Users/asus/Downloads/arena 2022.png",1)
    img=cv2.resize(img,(int(820/2),int(1050/2)))
    cv2.imshow("image",img)
    cv2.setMouseCallback("image",callback1)
    k=cv2.waitKey(0)
    if k==ord('q'):
        cv2.destroyAllWindows()
    if k==ord('r'):
        cv2.destroyAllWindows()
        show()

show()
