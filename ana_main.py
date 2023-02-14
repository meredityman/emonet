import numpy as np
from pathlib import Path
import argparse

import cv2

import zmq
import torch
import json
import time
from emonet.models import EmoNet

_expressions = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt'}

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
parser.add_argument('--cascPath', default="haarcascade_frontalface_default.xml")

parser.add_argument('--subIp'   , default="127.0.0.1")
parser.add_argument('--subPort' , default=8104)

args = parser.parse_args()

context = zmq.Context()

socket  = context.socket(zmq.PUB)
socket.bind(f"tcp://{args.subIp}:{args.subPort}")




class EmotionDetector():

    def __init__(self, device, n_expression):
        self.device = device

        state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        self.net = EmoNet(n_expression=n_expression).to(device)
        self.net.load_state_dict(state_dict, strict=False)
        self.net = self.net.half()
        self.net.eval()



    def get_emotions(self, face):
        data          = torch.tensor(face, dtype=torch.float16) / 255
        data          = data.permute(2, 0, 1).unsqueeze(0)

        images = data.to(self.device)
        with torch.no_grad():
            out = self.net(images)

        expression_logits = out['expression'].cpu().numpy()
        expr_i = int(np.argmax(expression_logits, axis=1).squeeze())

        val = out['valence']
        val = np.squeeze(val.cpu().numpy())

        ar = out['arousal']
        ar = np.squeeze(ar.cpu().numpy())

        expr = _expressions[expr_i]

        expressions = {}
        for i, emotion in _expressions.items():
            expressions[emotion] = float(expression_logits[0, i])



        return val, ar, expr, expressions

class FaceExtractor:

    def __init__(self, cascPath):
        self.faceCascade   = cv2.CascadeClassifier(cascPath)


    def get_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) >= 1:

            center_offset = []
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                centroid = np.array([x + w * 0.5, y + h * 0.5])
                offset = centroid - (np.array(frame.shape[:2]) * 0.5)
                offset = np.linalg.norm(offset)
                center_offset.append( offset )

            frame_i = np.argmin(np.array(center_offset))

            (x, y, w, h) = faces[frame_i]
            face =  frame[y:y+h, x:x+w]

            return face, (x, y, w, h)

        else:
            return None, (0, 0, 0, 0)

def main():
    inference_device = 'cuda:0'
    image_size = 256
    video_device = 0

    cascPath      = str(Path(args.cascPath).absolute())
    video_capture = cv2.VideoCapture(video_device)

    faceExtractor = FaceExtractor(cascPath)

    emotionDetector = EmotionDetector(device=inference_device, n_expression=8)


    t0 = time.time()
    while True:
        try:

            t0 = time.time()
            ret, frame = video_capture.read()

            if frame is not None:
                face, (x, y, w, h) = faceExtractor.get_face(frame)



                if face is not None:
                    t1 = time.time()
                    face  = cv2.cvtColor(face , cv2.COLOR_BGR2RGB)
                    face  = cv2.resize(face, (image_size, image_size))


                    (val, ar, expr, expressions) = emotionDetector.get_emotions(face)
                    t2 = time.time()

                    json_data = {
                        "valance"    : float(val),
                        "arousal"    : float(ar),
                        "expression" : expr,
                        "logits"     : expressions,
                        "box" : {
                            "x" : int(x),   
                            "y" : int(y),
                            "z" : int(w),
                            "w" : int(h)
                        },
                        "resolution" : {
                            "x" : int(frame.shape[1]),
                            "y" : int(frame.shape[2])
                        }
                    }

                    socket.send_string("ANA_Face", zmq.SNDMORE)
                    socket.send_string(json.dumps(json_data), 0)
                    t3 = time.time()

                    print(f"fps: {1/(t3-t0):02f} cv: {t1-t0:02f} | emonet: {t2-t1:02f} | zmq: {t3-t2:02f}")


                    # cv2.imshow('face', face)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break

            else:
                print(f"No camera frame!")
                time.sleep(1.)
                
        except KeyboardInterrupt:
            break


    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()