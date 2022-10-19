from enum import Enum
from operator import le
# Matrix manipulation
# stable and fast array processing library
import numpy as np

# process image
import PIL
from PIL import Image as im

# Importing the OpenCV library, CV2 support Numpy, Speed
import cv2


# class Analygraph(Enum):
#     true_analygraph = 1
#     gray_analygraph = 2
#     color_analygraph = 3
#     half_color_analygraph = 4
#     three_D_TV_optimized_analygraph = 5
#     DuBois_analygraph = 6
#     Roscolux_analygraph = 7

class Analygraph:
    def __init__(self, name, rgb_l, rgb_r):
        self.name = name
        self.leftrgb = rgb_l
        self.rightrgb = rgb_r
        self.rgb = 0
    def print_rgb(self):
        print(self.rgb)

class true_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "true_analygraph"

        self.Ml = np.array([[0.299, 0.587, 0.114], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0.299, 0.587, 0.114]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
   
class gray_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "gray_analygraph"

        self.Ml = np.array([[0.299, 0.587, 0.114], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0.299, 0.587, 0.114],
                    [0.299, 0.587, 0.114]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
    
class color_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "color_analygraph"

        self.Ml = np.array([[1, 0, 0], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
    
class color_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "color_analygraph"

        self.Ml = np.array([[1, 0, 0], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
    
class half_color_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "half_color_analygraph"

        self.Ml = np.array([[0.299, 0.587, 0.114], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))

class three_D_TV_optimized_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "three_D_TV_optimized_analygraph"

        self.Ml = np.array([[0, 0.7, 0.3], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
    
class Roscolux_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "Roscolux_analygraph"

        self.Ml = np.array([[0.3185, 0.0769, 0.0109], 
                    [0.1501, 0.0767, 0.0056],
                    [0.0007, 0.0020, 0.0156]])

        self.Mr = np.array([[0.0174, 0.0484, 0.1402],
                    [0.0184, 0.1807, 0.0458],
                    [0.0268, 0.0991, 0.7662]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
 
class DuBois_analygraph(Analygraph):
    def __init__(self, name, rgb_l, rgb_r):
        super().__init__(name, rgb_l, rgb_r)
        
        self.type = "DuBois_analygraph"

        self.Ml = np.array([[0.437, 0.449, 0.164], 
                    [-0.062, -0.062, -0.024],
                    [-0.048, -0.050, -0.017]])

        self.Mr = np.array([[-0.011, -0.032, -0.007],
                    [0.377, 0.761, 0.009],
                    [-0.026, -0.093, 1.234]])

        self.rgb = np.add(np.matmul(self.leftrgb, self.Ml), np.matmul(self.rightrgb, self.Mr))
    
def rgb(analygraph, l_rgb, r_rgb):
    return np.add(np.matmul(analygraph.Ml,l_rgb), np.matmul(analygraph.Mr, r_rgb))

def get_analygraph(left_img,right_img, analygraph_type, out):
    for i in range(left_img.shape[0]):
                for j in range(left_img.shape[1]):
                    out[i,j] = rgb(analygraph_type, left_img[i,j], right_img[i,j])

def split_image(image, orient):
    joined_img = image
    h, w = joined_img.shape[:2]
    
    # divide image by width
    if(orient == 1):
        mid = w//2
    elif(orient == 2):
        mid = h//2

    # left image
    left_part = joined_img[:, :mid] 
    
    # right image
    right_part = joined_img[:, mid:]  

    return left_part, right_part





def main():
    # Reading the image using imread() function
    image = cv2.imread('image.jpeg')
    
    # Extracting the height and width of an image
    h, w = image.shape[:2]
    # Displaying the height and width
    print("Height = {},  Width = {}".format(h, w))

    l, r = split_image(image,1)

    # save image
    status = cv2.imwrite('/Users/tanxinkai/Desktop/Analygraph_Project/left_image.jpeg',l) 
    print("Image written to file-system : ",status)
    status = cv2.imwrite('/Users/tanxinkai/Desktop/Analygraph_Project/right_image.jpeg',r)
    print("Image written to file-system : ",status)


    h, w = l.shape[:2]
    print(h,w)

    # initialise new image
    ans = np.zeros((h,w,3), np.uint8)

    # print(f"l.shape[0] is {l.shape[0]}")
    # print(f"l.shape[1] is {l.shape[1]}")

    # print(f"r.shape[0] is {r.shape[0]}")
    # print(f"r.shape[1] is {r.shape[1]}")

    test_rgbL = np.array([[1,1,250]])
    test_rgbR = np.array([[0,0,255]])
    a = true_analygraph("a", test_rgbL, test_rgbR)
    b = color_analygraph("b", test_rgbL, test_rgbR)
    c = gray_analygraph("c", test_rgbL, test_rgbR)
    d = half_color_analygraph("d", test_rgbL, test_rgbR)
    e = three_D_TV_optimized_analygraph("e", test_rgbL, test_rgbR)
    f = DuBois_analygraph("f", test_rgbL, test_rgbR)
    g = Roscolux_analygraph("g", test_rgbL, test_rgbR)
    
    pixel_arr = [a,b,c,d,e,f,g]

    a.print_rgb()
    b.print_rgb()
    
    tries = 3

    while(tries):
        print("[-1]" + "   " + "Exit")
        for x in range(len(pixel_arr)):
            print("[" + str(x) + "]" + "   " + pixel_arr[x].type)
        choice = int(input('Choose a type \n'))
        if(choice >= 0 and choice <= 6):
            val = pixel_arr[choice]

            # fill image by pixels
            get_analygraph(l, r, val, ans)
           
    
            status = cv2.imwrite('/Users/tanxinkai/Desktop/Analygraph_Project/analygraph.jpeg',ans) 
            print("Image written to file-system : ",status)
        elif(choice == -1):
            break
        else:
            print("Choice out of range")
        tries-=1
    
    print("exiting program")
    
if __name__ == "__main__":
        main()