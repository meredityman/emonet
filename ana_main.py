import numpy as np
from pathlib import Path
import argparse
from threading import Thread
import cv2

import zmq
import torch
import json
import time
from emonet.models import EmoNet




#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=5, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
parser.add_argument('--cascPath', default="haarcascade_frontalface_default.xml")
parser.add_argument('--device'  , type=int, default=0)
parser.add_argument('--subIp'   , default="0.0.0.0")
parser.add_argument('--subPort' , default=8104)

args = parser.parse_args()

context = zmq.Context()

socket  = context.socket(zmq.PUB)
socket.bind(f"tcp://{args.subIp}:{args.subPort}")

if args.nclasses == 8:
    _expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt'}
elif args.nclasses == 5: 
    _expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear'}


class FaceExtractor:

    def __init__(self, cascPath):
        print("Initialising face extractor")
        self.faceCascade   = cv2.CascadeClassifier(cascPath)
        self.face = None
        self.box = None

    def ready(self):
        pass

    def get_face(self, frame):
        self.frame = frame
        newFace = self._get_face()
        return self.face, self.box, newFace
    
    def _get_face(self):
        frame = self.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) >= 1:
            if len(faces) > 1: 
                print(f"Warning: Multpile faces")

            metric = []
            for face in faces:
                if(self.box == None):
                    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    (x, y, w, h) = face
                    centroid = np.array([x + w * 0.5, y + h * 0.5])
                    offset = (np.array(frame.shape[:2]) * 0.5) - centroid
                    offset = np.linalg.norm(offset)
                    metric.append( offset )
                else:
                    offset = np.linalg.norm(np.array(face) - np.array(self.box))
                    metric.append( offset )

            frame_i = np.argmin(np.array(metric))

            (x, y, w, h) = faces[frame_i]
            face =  frame[y:y+h, x:x+w]

            self.face = face
            self.box  = (x, y, w, h)
            return True
        else:
            return False


class EmotionDetector():

    def __init__(self, device, n_expression):
        print("Initializing emotion detector")
        self.device = device

        state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        print("Loading network...")
        self.net = EmoNet(n_expression=n_expression).half().to(device)
        print("Loading state dict...")
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()

        self.face  = None
        self.thread = None

        self.val         = None
        self.ar          = None
        self.expr        = None
        self.expressions = None

    def get_emotions(self, face):
        
        if(self.thread is None or not self.thread.is_alive()):
            self.face = face
            self.thread = Thread(target=self.get_emotions_thread)
            self.thread.start()

        return (self.val, self.ar, self.expr, self.expressions)

    def get_emotions_thread(self):
        face = self.face
        data          = torch.tensor(face, dtype=torch.float16) / 255
        data          = data.permute(2, 0, 1).unsqueeze(0)

        images = data.to(self.device)
        with torch.no_grad():
            out = self.net(images)

        expression_logits = out['expression'].cpu().numpy()
        expression_logits =   (expression_logits / np.sqrt(np.sum(expression_logits**2)))
        expression_logits = np.clip(0.5 *  (expression_logits + 1.0), 0.0, 1.0)
        expr_i = int(np.argmax(expression_logits, axis=1).squeeze())    

        val = out['valence']
        val = np.squeeze(val.cpu().numpy())

        ar = out['arousal']
        ar = np.squeeze(ar.cpu().numpy())

        expr = _expressions[expr_i]

        expressions = {}
        for i, emotion in _expressions.items():
            expressions[emotion] = float(expression_logits[0, i])

        self.val  = val
        self.ar   = ar
        self.expr = expr
        self.expressions = expressions

def main():
    inference_device = 'cuda:0'
    image_size = 256
    
    cascPath      = str(Path(args.cascPath).absolute())
    video_capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)

    faceExtractor = FaceExtractor(cascPath)

    emotionDetector = EmotionDetector(device=inference_device, n_expression=args.nclasses)

    t0 = time.time()
    while True:
        try:
            t0 = time.time()
            ret, frame = video_capture.read()

            if frame is not None:
                face, (x, y, w, h), newFace = faceExtractor.get_face(frame)
                t1 = time.time()

                if face is not None and newFace:
                    # face  = cv2.cvtColor(face , cv2.COLOR_BGR2RGB)
                    face  = cv2.resize(face, (image_size, image_size))

                    (val, ar, expr, expressions) = emotionDetector.get_emotions(face)
                    t2 = time.time()

                    json_data = {
                        "valance"    : float(val) if val  else 0.0,
                        "arousal"    : float(ar)  if ar   else 0.0,
                        "expression" : expr       if expr else None,
                        "expression-probability" : 100.0 * expressions[expr] if expr else None,
                        "logits"     : expressions,
                        "box" : {
                            "x" : int(x),   
                            "y" : int(y),
                            "z" : int(w),
                            "w" : int(h)
                        },
                        "resolution" : {
                            "x" : int(frame.shape[1]),
                            "y" : int(frame.shape[0])
                        },
                    }

                    socket.send_string("ANA_Face", zmq.SNDMORE)
                    socket.send_string(json.dumps(json_data), 0)
                    t3 = time.time()

                    print(json_data)
                    print(f"fps: {1/(t3-t0):02f} cv: {t1-t0:02f} | emonet: {t2-t1:02f} | zmq: {t3-t2:02f}  | exp: {expr}")


                    cv2.imshow('face', face)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                print(f"No camera frame!")
                time.sleep(1.)
                
        except KeyboardInterrupt:
            break


    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()