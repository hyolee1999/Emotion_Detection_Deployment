# from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64


# def pil_image_to_base64(pil_image):
#     buf = BytesIO()
#     pil_image.save(buf, format="JPEG")
#     return base64.b64encode(buf.getvalue())


# def base64_to_pil_image(base64_img):
#     return Image.open(BytesIO(base64.b64decode(base64_img)))
def opencv_to_base64(cv_image):

    _, im_arr = cv2.imencode('.jpg', cv_image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes)
    # buf = BytesIO()
    # cv_image.save(buf, format="JPEG")
    # return base64.b64encode(buf.getvalue())

def base64_to_opencv(base64_img):
   
    im_bytes = base64.b64decode(base64_img)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # return Image.open(BytesIO(base64.b64decode(base64_img)))

