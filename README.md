# LPR_in_colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1hMbVFzHl7C7d_A3q6dEHcCb-ElK4MzkN/view?usp=sharing)  (Executing with GPU)<br>
Using morphological image processing and easyOCR to recognize license plates; Demonstrating the realtime detection with webcam in Google Colab.<br>
* **colab_LPR.ipynb** demonstrates the execution in Google Colab
* **anpr_easy.py** ([reference](https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/)) and **colab_cam.py** ([reference](https://github.com/OmniXRI/Colab_Webcam_OpenCV)) are files needed to be uploaded in Colab when we execute colab_LPR.ipynb

## License Plates Recognition
This question can be divided into two parts: object detection and optical character recognition (OCR). 
* Object detection<br>
I use morphological image processing to detect the license plates according to the article on PyImageSearch ([reference](https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/)). In the article, the author uses blackhat morphological operation to reveals dark characters against light backgrounds. Then, Sobel operator, blurring, and closing are used to find boundaries of the characters and fill the holes, which locates the characters. Next, after a series of iterative erosions and dilations, the author uses light regions as a mask to reveal the license plate candidates. The author restricts the aspect ratios of the candidates since the aspect ratio are usually in certain interval. To accelerate the filtering process, I add one restriction regarding the lower limit of $\frac{\text{Area of Contours}}{\text{Area of Bounding Box}}$, which filters out the candidates on messy backgrounds.<br>
*Note: In real life the restriction of y coordinate can be useful since the license plates are close to the ground.
* OCR<br>
I use `easyOCR` to perform OCR, specifying the allow list of the result. The `easyocr.Reader(['en'])` is not written in **anpr_easy.py** because it takes a while to download the detection model.

## Webcam and Realtime License Plates Recognition in Google Colab
According to Colab_Webcam_OpenCV repository ([reference](https://github.com/OmniXRI/Colab_Webcam_OpenCV)), I edit the `realtime_process()` function to draw LPR results on the screen.<br>
The execution in Google Colab is `main.ipynb`. Demo video is as below...<br>
*Note: `anpr_easy.py` and `colab_cam.py` must be uploaded to the current Colab environment before we run `main.ipynb`.<br>
<br>
![image](https://github.com/105304039/LPR_in_colab/blob/main/demo/Executiion%20in%20colab.gif)
