
import numpy as np
import cv2


face_classifier = "haarcascades/haarcascade_frontalface_alt.xml"
eyes_classifier = "haarcascades/haarcascade_eye.xml"
eye_left_classifier = "haarcascades/haarcascade_mcs_lefteye.xml"
eye_right_classifier = "haarcascades/haarcascade_mcs_righteye.xml"
nose_classifier = "haarcascades/haarcascade_mcs_nose.xml"
mouth_classifier = "haarcascades/haarcascade_mcs_mouth.xml"
ear_left_classifier = "harcassades/haarcascade_mcs_leftear.xml"
ear_right_classifier = "harcassades/haarcascade_mcs_leftear.xml"

face_cascade = cv2.CascadeClassifier(face_classifier)
eyes_cascade = cv2.CascadeClassifier(eyes_classifier)
eye_left_cascade = cv2.CascadeClassifier(eye_left_classifier)
eye_right_cascade = cv2.CascadeClassifier(eye_right_classifier)
nose_cascade = cv2.CascadeClassifier(nose_classifier)
mouth_cascade = cv2.CascadeClassifier(mouth_classifier)
ear_left_cascade = cv2.CascadeClassifier(ear_left_classifier)
ear_right_cascade = cv2.CascadeClassifier(ear_right_classifier)

colors = {
    "red": (255,0,0),
    "green": (0,255,0),
    "blue": (0,0,255),
    "yellow": (255,255,0),
    "cyan": (0,255,255),
    "magenta": (255,0,255)
}

class FaceError(Exception):
    def __init__(self, message):
        self.message = message

def read_image(path):
    image = cv2.imread(path, cv2.CV_LOAD_IMAGE_UNCHANGED)
    height, width, channels = image.shape
    if channels < 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    return image

def overlay_images(background, foreground, x, y):
    fg_height, fg_width, fg_channels = foreground.shape
    for c in range(0,3):
        background[max(0,y-fg_height):y, x:x+fg_width, c] = (foreground[max(0,fg_height-y):fg_height, :, c]) * (foreground[max(0,fg_height-y):fg_height, :, 3]/255.0) + (background[max(0,y-fg_height):y, x:x+fg_width, c]) * (1.0 - foreground[max(0,fg_height-y):fg_height, :, 3]/255.0)

class Rectangle:
    def __init__(self, values):
        self.x = values[0]
        self.y = values[1]
        self.width = values[2]
        self.height = values[3]

    def __str__(self):
        return "x: {0}\ny: {1}\nwidth: {2}\nheight: {3}\n".format(self.x, self.y, self.width, self.height)

    def set_offset(self, offsets):
        self.x = self.x + offsets.x
        self.y = self.y + offsets.y

    def draw_to_image(self, image, color):
        cv2.rectangle(image,(self.x, self.y), (self.x + self.width, self.y + self.height), color, 2)
        return image

class FaceEditor:
    def __init__(self, image_in, is_matrix=False):
        self.image_path = image_in
        try:
            self.image = image_in if is_matrix else read_image(image_in)
            self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.face = self.find_face()
            self.face_grayscale = self.grayscale[self.face.y:self.face.y+self.face.height, self.face.x:self.face.x+self.face.width]
            self.eye_right = self.find_eye_right()
            self.eye_left = self.find_eye_left()
            self.nose = self.find_nose()
            self.mouth = self.find_mouth()
            self.ear_left = None
            self.ear_right = None
        except Exception:
            raise FaceError("Error finding face")
        # Overlays
        self.hat = None
        self.moustache = None

    def find_face(self):
        results = face_cascade.detectMultiScale(self.grayscale, 1.3, 5)
        return Rectangle(face_cascade.detectMultiScale(self.grayscale, 1.3, 5)[0])

    def find_eye_left(self):
        # Ensure mouth is is bottom half of face
        result_iterator = 0
        results = eye_left_cascade.detectMultiScale(self.face_grayscale)
        while result_iterator < len(results):
            current_result = Rectangle(results[result_iterator])
            # If result is in top left quadrant of face we found result
            if current_result.y < self.face.height / 2 and current_result.x > self.face.width / 3:
                break
            result_iterator = result_iterator + 1
        current_result.set_offset(self.face)
        return current_result

    def find_eye_right(self):
        # Ensure mouth is is bottom half of face
        result_iterator = 0
        results = eye_right_cascade.detectMultiScale(self.face_grayscale)
        while result_iterator < len(results):
            current_result = Rectangle(results[result_iterator])
            # If result is in top left quadrant of face we found result
            if current_result.y < self.face.height / 2 and current_result.x < self.face.width / 3:
                break
            result_iterator = result_iterator + 1
        current_result.set_offset(self.face)
        return current_result

    def find_nose(self):
        res = Rectangle(nose_cascade.detectMultiScale(self.face_grayscale)[0])
        res.set_offset(self.face)
        return res

    def find_mouth(self):
        # Ensure mouth is is bottom half of face
        result_iterator = 0
        results = mouth_cascade.detectMultiScale(self.face_grayscale)
        while result_iterator < len(results):
            current_result = Rectangle(results[result_iterator])
            # If result is in bottom half of face we found mouth
            if current_result.y > self.face.height / 2:
                break
            result_iterator = result_iterator + 1
        current_result.set_offset(self.face)
        return current_result

    def draw_feature_boxes(self):
        result = self.face.draw_to_image(self.image, colors["red"])
        result = self.eye_right.draw_to_image(self.image, colors["green"])
        result = self.eye_left.draw_to_image(self.image, colors["green"])
        result = self.nose.draw_to_image(self.image, colors["blue"])
        result = self.mouth.draw_to_image(self.image, colors["magenta"])
        return result

    def overlay_hat(self, hat):
        if hat is None:
            self.hat = None
            return
        # Read hat image
        image = read_image(hat)
        # Resize hat
        height, width, channels = image.shape
        scale = float(float(width) / float(self.face.width))
        new_size =  int(float(width) / scale), int(float(height) / scale)
        # Resize the hat
        self.hat = cv2.resize(src=image, dsize=new_size, interpolation=cv2.INTER_AREA)

    def overlay_moustache(self, moustache):
        if moustache is None:
            self.moustache = None
            return
        # Read moustache image
        image = read_image(moustache)
        # Resize hat
        height, width, channels = image.shape
        scale = float(float(width) / float(self.face.width))
        new_size =  int(float(width) / scale), int(float(height) / scale)
        # Resize the moustache
        self.moustache = cv2.resize(src=image, dsize=new_size, interpolation=cv2.INTER_AREA)

    def draw_image(self):
        # Default iamge is basic image
        result = self.image
        # Draw moustache
        if self.moustache is not None:
            overlay_images(result, self.moustache, self.nose.x-(self.moustache.shape[1]/2)+(self.nose.width/2), (self.mouth.y+self.nose.y+self.nose.height)/2 + 20)
        # Draw hat
        if self.hat is not None:
            y_offset = min(self.eye_right.y, self.eye_left.y) + self.face.height/3
            overlay_images(result, self.hat, self.face.x, y_offset)
        return result

image_name = '/Users/saad/Desktop/saad.jpg'
hat_name = '/Users/saad/Desktop/nick/face/overlays/hats/batman.png'
moustache_name = '/Users/saad/Desktop/nick/face/overlays/moustache.png'


import sys

mode = "squares"
if len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode == "help":
        print("Modes are: squares batman help")
        exit()


cv2.namedWindow("Batman", flags=cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
while True:
    ret, webcam = capture.read()
    if ret:
        try:
            result = FaceEditor(webcam, True)
        except FaceError as e:
            print("Error finding face")
            continue
        if mode == "batman":
            result.overlay_moustache(moustache_name)
            result.overlay_hat(hat_name)
            cv2.imshow("Batman", result.draw_image())
        if mode == "squares":
            result = result.draw_feature_boxes()
            cv2.imshow("Batman", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
