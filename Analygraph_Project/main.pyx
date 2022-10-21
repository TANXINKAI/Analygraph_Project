# Matrix manipulation
# stable and fast array processing library
import numpy as np

# process image
# Importing the OpenCV library, CV2 support Numpy, Speed
import cv2

cdef class Analygraph:
    def __init__(self, name):
        self.name = name
   

cdef class true_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)
        
        self.type = "true_analygraph"

        self.Ml = np.array([[0.299, 0.587, 0.114], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0.299, 0.587, 0.114]])

cdef class gray_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)
        
        self.type = "gray_analygraph"

        self.Ml = np.array([[0.299, 0.587, 0.114], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0.299, 0.587, 0.114],
                    [0.299, 0.587, 0.114]])

cdef class color_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)
        
        self.type = "color_analygraph"

        self.Ml = np.array([[1, 0, 0], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

cdef class half_color_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)

        self.type = "half_color_analygraph"

        self.Ml = np.array([[0.299, 0.587, 0.114], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

cdef class three_D_TV_optimized_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)
        
        self.type = "three_D_TV_optimized_analygraph"

        self.Ml = np.array([[0, 0.7, 0.3], 
                    [0, 0, 0],
                    [0, 0, 0]])

        self.Mr = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

      
cdef class Roscolux_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)
        
        self.type = "Roscolux_analygraph"

        self.Ml = np.array([[0.3185, 0.0769, 0.0109], 
                    [0.1501, 0.0767, 0.0056],
                    [0.0007, 0.0020, 0.0156]])

        self.Mr = np.array([[0.0174, 0.0484, 0.1402],
                    [0.0184, 0.1807, 0.0458],
                    [0.0268, 0.0991, 0.7662]])

       
cdef class DuBois_analygraph(Analygraph):
    def __init__(self, name):
        super().__init__(name)
        
        self.type = "DuBois_analygraph"

        self.Ml = np.array([[0.437, 0.449, 0.164], 
                    [-0.062, -0.062, -0.024],
                    [-0.048, -0.050, -0.017]])

        self.Mr = np.array([[-0.011, -0.032, -0.007],
                    [0.377, 0.761, 0.009],
                    [-0.026, -0.093, 1.234]])

       
cpdef rgb(analygraph, l_rgb, r_rgb):
    return np.add(np.matmul(analygraph.Ml,l_rgb), np.matmul(analygraph.Mr, r_rgb))

cpdef void get_analygraph(unsigned char [:, :] left_img, unsigned char [:, :] right_img, unsigned char [:, :, :] out, Analygraph analygraph_type):
# def get_analygraph(left_img,right_img, analygraph_type, out):
   
    # set the variable extension types
    cdef int i, j, w, h
    h = left_img.shape[0]
    w = left_img.shape[1]

    for i in range(h):
                for j in range(w):
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
    image = cv2.imread('/Users/tanxinkai/Desktop/Analygraph_Project/Analygraph_Project/image.jpeg')
    
    h, w = image.shape[:2]

    # Display height and width
    # print("Height = {},  Width = {}".format(h, w))

    l, r = split_image(image,1)

    # save image
    left_status = cv2.imwrite('/Users/tanxinkai/Desktop/Analygraph_Project/Analygraph_Project/left_image.jpeg',l) 
    print("Image saved :  : ",left_status)
    right_status = cv2.imwrite('/Users/tanxinkai/Desktop/Analygraph_Project/Analygraph_Project/right_image.jpeg',r)
    print("Image saved :  : ",right_status)


    h, w = l.shape[:2]
    print(h,w)

    # initialise new image
    ans = np.zeros((h,w,3), np.uint8)

    # print(f"l.shape[0] is {l.shape[0]}")
    # print(f"l.shape[1] is {l.shape[1]}")

    # print(f"r.shape[0] is {r.shape[0]}")
    # print(f"r.shape[1] is {r.shape[1]}")

    # test_rgbL = np.array([[1,1,250]])
    # test_rgbR = np.array([[0,0,255]])
    a = true_analygraph("a")
    b = color_analygraph("b")
    c = gray_analygraph("c")
    d = half_color_analygraph("d")
    e = three_D_TV_optimized_analygraph("e")
    f = DuBois_analygraph("f")
    g = Roscolux_analygraph("g")
    
    pixel_arr = [a,b,c,d,e,f,g]

    tries = 3

    while(tries):
        print("[-1]" + "   " + "Exit")
        for x in range(len(pixel_arr)):
            print("[" + str(x) + "]" + "   " + pixel_arr[x].type)
        choice = int(input('Choose a type \n'))
        if(choice >= 0 and choice <= 6):
            val = pixel_arr[choice]

            # fill image by pixels
            get_analygraph(l, r, ans, val)
           
    
            status = cv2.imwrite('/Users/tanxinkai/Desktop/Analygraph_Project/analygraph.jpeg',ans) 
            print("Image saved :  : ",status)
        elif(choice == -1):
            break
        else:
            print("Choice out of range")
        tries-=1
    
    print("exiting program")
    
if __name__ == "__main__":
        main()