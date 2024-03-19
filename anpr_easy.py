#  Implementing ANPR/ALPR with OpenCV and Python (Automatic number-plate recognition)
# https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
# closing連結破孔/消除雜訊；opening去除小砸點，銳利邊界。(https://medium.com/%E9%9B%BB%E8%85%A6%E8%A6%96%E8%A6%BA/%E5%BD%A2%E6%85%8B%E5%AD%B8-morphology-%E6%87%89%E7%94%A8-3a3c03b33e2b)

# import the necessary packages
from skimage.segmentation import clear_border # this method assists with cleaning up the borders of images.

import numpy as np
import imutils
import cv2
import easyocr
# from adjust import contour_fix


class PyImageSearchANPR:
    def __init__(self,reader,minAR=4, maxAR=5, debug=False, license_num = 7,verbose = False, keep = 5,contour_box_ratio=0):
        self.minAR = minAR  #  最小長寬比(aspect ratio = 寬/高): 用來偵測/過濾長方形車牌 (default = 4)
        self.maxAR = maxAR  #  最大長寬比(aspect ratio = 寬/高): 用來偵測/過濾長方形車牌 (default = 5)
        self.debug = debug  #  是否在debug mode
        self.license_num = license_num
        self.verbose = verbose
        self.keep = keep
        self.reader = reader
        self.contour_box_ratio = contour_box_ratio # [ contour 佔 boundingRect 比例]條件 理想是1，0表示不篩選

    def locate_license_plate_candidates(self, gray):

        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))   # blackhat morphological operation: kernel是長方形 13 pixels wide x 5 pixels tall, 對應一般國際車牌形狀
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern) # blackhat morphological operation

        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)    # 用closing連結「分離但又相去不遠的區塊」，又或是「填滿影像中的破孔」或「消除雜訊」，因為先做Dilation再做erosion 
        light = cv2.threshold(light, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]                    # binary threshold用Otsu’s method顯示light regions in the image that may contain license plate characters

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
            dx=1, dy=0, ksize=-1)                         # 把blackhat中的字元描出框，變gradient magnitude image 
        gradX = np.absolute(gradX)                        # 然後scale the resulting intensities至[0,255]
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")

        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)        # 對 gradient magnitude image 做高斯模糊
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern) # 填補孔洞
        thresh = cv2.threshold(gradX, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]       # Otsu's method二值化         

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        thresh = cv2.bitwise_and(thresh, thresh, mask=light) # 把light當作遮罩
        thresh = cv2.dilate(thresh, None, iterations=2)      # 再經過一連串dilate erode降噪....
        thresh = cv2.erode(thresh, None, iterations=1)


        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)                       #  輸入圖像是二值，黑色背景，白色目標；輪廓檢索方式 ；輪廓近似方法。輸出兩個值：contours, hierarchy
        cnts = imutils.grab_contours(cnts) # 找出不同版本(opencv 2 or 3)的返回contours


        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:self.keep]
        cnts = sorted(cnts, key=self.sort_c_on_bbox, reverse=True)
        # return the list of contours
        return cnts

    def sort_c_on_bbox(self,c):
        _,_,w,h = cv2.boundingRect(c)   
        return cv2.contourArea(c)/(w*h)

    def sharpen0(self,img, gauss_kernel = 0,sigma=100,org_wt = 1.5,blur_wt = -0.5,add_constant = 0):    
        """銳化圖片"""
        blur_img = cv2.GaussianBlur(img, (gauss_kernel, gauss_kernel), sigma)
        usm = cv2.addWeighted(img, org_wt, blur_img, blur_wt, add_constant) 
        return usm


    def locate_license_plate(self, gray, candidates,clearBorder=False):  # clearBorder = 是否需要剔除碰到照片邊框的輪廓?
        # initialize the license plate contour and ROI
        lpCnt = None
        roi = None
        cand_list = []

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if ar >= self.minAR and ar <= self.maxAR and (cv2.contourArea(c)/(w*h))>self.contour_box_ratio:  # 新增[ contour 佔 boundingRect 比例]條件?
                lpCnt = c                                       # current contour 
                roi = gray[y:y + h, x:x + w]
                cand_list.append([roi, lpCnt])

        return cand_list



    def edited(self,gray,border_ratio = 0.15,ratio = 3):
        e1 = cv2.copyMakeBorder(gray,0,int(border_ratio*gray.shape[0]),0,0,cv2.BORDER_REPLICATE)
        return cv2.resize(e1,(e1.shape[1],int(e1.shape[1]/ratio)))


    def find_and_ocr(self, image, psm=7, clearBorder=False):
        lpText = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        cand_list = self.locate_license_plate(gray, candidates,clearBorder=clearBorder)


        for i,(lp, lpCnt) in enumerate(cand_list):
            if lp is not None:
                # OCR the license plate
                res = self.reader.readtext(lp, allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",detail = 0)
                if res!=[]:
                  lpText = res[0] 
                  if len(lpText)>=self.license_num:
                      if self.verbose in [0,1,2,3]:
                          print(i,"Finish")
                      return (lpText, lpCnt)
        return #cand_list