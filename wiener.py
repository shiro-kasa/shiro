import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2
import os

#仿真運動模糊

def motion_process(image_size,motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0]-1)/2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1/ slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i* slope_tan) # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)]=1
        return PSF / PSF.sum()
    else:
        for i in range(15):
            offset = round(i*slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)]=1
        return PSF / PSF.sum()

#對圖片僅行運動模糊
def make_blurred(input,PSF,eps):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred

def inverse(input,PSF,eps):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    result = fft.ifft2(input_fft / PSF_fft)
    result = np.abs(fft.fftshift(result))
    return result

def wiener(input,PSF,eps, K=0.001):     #維納濾波 K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result

def normal(array):
    array = np.where(array <0, 0, array)
    array = np.where(array > 255, 255, array)
    array = array.astype(np.int16)
    return array

def main(gray):
    channel = []
    img_h, img_w = gray.shape[:2]
    PSF = motion_process((img_h,img_w), 60)   #進行運動模糊處理
    blurred = np.abs(make_blurred(gray, PSF, 1e-3))

    result_blurred = inverse(blurred, PSF, 1e-3)  #逆濾波
    result_wiener = wiener(blurred, PSF, 1e-3)    #維納濾波

    blurred_noisy = blurred +0.1 * blurred.std() * \
                    np.random.standard_normal(blurred.shape)  #添加噪聲,standard_normal產生隨機的函數
    inverse_mo2no = inverse(blurred_noisy, PSF, 0.1 + 1e-3)   #對添加噪聲的圖像進行逆濾波
    wiener_mo2no = wiener(blurred_noisy, PSF, 0.1 + 1e-3)     #對添加噪聲的圖像進行維納濾波
    channel.append((normal(blurred),normal(result_blurred),normal(result_wiener),normal(blurred_noisy),normal(inverse_mo2no),normal(wiener_mo2no)))
    
    return channel

if __name__ == '__main__':

    path="../datasets/papper_db-5/train/images"
    data_path="../datasets/papper_db-5/train/images"

    if os.path.isdir(path):
        print("目錄存在。")
    else:
        os.mkdir(path)

    for filename in os.listdir(data_path):
        print(filename) #just for test
        #img is used to store the image data 
        image = cv2.imread(data_path + "/" + filename)
        b_gray, g_gray, r_gray = cv2.split(image.copy())

        Result = []
        for gray in [b_gray, g_gray, r_gray]:
            channel = main(gray)
            Result.append(channel)
        
        blurred = cv2.merge([Result[0][0][0], Result[1][0][0], Result[2][0][0]])
        result_blurred = cv2.merge([Result[0][0][1], Result[1][0][1], Result[2][0][1]])
        result_wiener = cv2.merge([Result[0][0][2], Result[1][0][2], Result[2][0][2]])
        blurred_noisy = cv2.merge([Result[0][0][3], Result[1][0][3], Result[2][0][3]])
        inverse_mo2no = cv2.merge([Result[0][0][4], Result[1][0][4], Result[2][0][4]])
        wiener_mo2no = cv2.merge([Result[0][0][5], Result[1][0][5], Result[2][0][5]])

        cv2.imwrite(path+"/"+filename,result_wiener)



    # image = cv2.imread('Dopamin01.jpg')
    # b_gray, g_gray, r_gray = cv2.split(image.copy())

    # Result = []
    # for gray in [b_gray, g_gray, r_gray]:
    #     channel = main(gray)
    #     Result.append(channel)
    
    # blurred = cv2.merge([Result[0][0][0], Result[1][0][0], Result[2][0][0]])
    # result_blurred = cv2.merge([Result[0][0][1], Result[1][0][1], Result[2][0][1]])
    # result_wiener = cv2.merge([Result[0][0][2], Result[1][0][2], Result[2][0][2]])
    # blurred_noisy = cv2.merge([Result[0][0][3], Result[1][0][3], Result[2][0][3]])
    # inverse_mo2no = cv2.merge([Result[0][0][4], Result[1][0][4], Result[2][0][4]])
    # wiener_mo2no = cv2.merge([Result[0][0][5], Result[1][0][5], Result[2][0][5]])

    # #==========可視化===========
    # plt.figure(1)
    # plt.xlabel("Original Image")
    # plt.imshow(np.flip(image, axis=2))   #顯示原圖像
    

    # plt.figure(2)
    # plt.figure(figsize=(8, 6.5))
    # imgNames = {"Motion blurred":blurred,
    #             "inver deblurred":result_blurred,
    #             "wiener deblurred(k=0.01)":result_wiener,
    #             "motion & noisy blurred":blurred_noisy,
    #             "inverse_mo2no": inverse_mo2no,
    #             "wiener_mo2no":wiener_mo2no}
    # for i,(key,imgNames) in enumerate(imgNames.items()):
    #     plt.subplot(231+i)
    #     plt.xlabel(key)
    #     plt.imshow(np.flip(imgNames,axis=2))
    
    # plt.figure(4)
    # plt.imshow(np.flip(result_wiener, axis=2)) 

    # plt.show()