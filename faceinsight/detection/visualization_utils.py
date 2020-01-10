# vi: set ft=python sts=4 ts=4 sw=4 et:

from PIL import ImageDraw
import numpy as np

def show_bboxes(img, bounding_boxes, facial_landmarks = []):
    """Draw bounding boxes and  facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'white')

    inx = 0
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[ i] + 1.0, p[i  + 5] + 1.0)
            ], outline = 'blue' )

    return img_copy 

def show_grids(img, bounding_boxes, facial_landmarks=[], step=1):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a  float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'white')

    inx = 0
    for pp in facial_landmarks:
        p = pp.reshape(2,5).T
        p = p.tolist()
        mouth_center = [(p[3][0] + p[4][0])/2, (p[3][1] + p[4][1])/2]
        eye_center = [(p[0][0] + p[1][0])/2, (p[0][1] + p[1][1])/2]
        p6 = [(p[2][0] - mouth_center[0])/4 + mouth_center[0], (p[2][1] - mouth_center[1])/4 + mouth_center[1]]
        p9 = [p[3][0] - (p[4][0]-p[3][0])/3 ,p[3][1] - (p[4][1]-p[3][1])/3]
        p10 = [p[4][0] + (p[4][0]-p[3][0])/3 ,p[4][1] + (p[4][1]-p[3][1])/3]
        p11 = [mouth_center[0] - (eye_center[0] - mouth_center[0])/2,mouth_center[1] - (eye_center[1] - mouth_center[1])/2]
        p12 = [(eye_center[0] -mouth_center[0])/4 + eye_center[0], (eye_center[1] - mouth_center[1])/4 + eye_center[1]]
        p13 = [(p[0][0] + p[3][0])/2,(p[0][1] + p[3][1])/2]
        p14 = [(p[1][0] + p[4][0])/2,(p[1][1] + p[4][1])/2]


        p.append(p6)
        p.append([p[0][0]-3/8*(p[1][0]-p[0][0]),3/2*p[0][1]-1/2*p[1][1]]) 
        p.append([p[1][0]+3/8*(p[1][0]-p[0][0]),3/2*p[1][1]-1/2*p[0][1]])
        p.append(p9)
        p.append(p10)
        p.append(p11)        
        p.append(p12)
        p.append(p13)
        p.append(p14)


        #for i in range(12):
        #    draw.ellipse([
        #         (p[i][0]-2.0,p[i][1]-2.0),
        #         (p[i][0]+2.0,p[i][1]+2.0)
        #     ],outline='white',fill='white')

        if step==1:
            draw.line(
                ((p[11][0],p[11][1]),
                 (p[7][0],p[7][1]),
                 (p[9][0],p[9][1]),
                 (p[10][0],p[10][1]),
                 (p[8][0],p[8][1]),
                 (p[6][0],p[6][1]),
                 (p[11][0],p[11][1])),
                fill=(136,232,232),
                width=1 
            )
            draw.line(
                ((p[11][0],p[11][1]),
                 (p[1][0],p[1][1]),
                 (p[2][0],p[2][1]),
                 (p[5][0],p[5][1]),
                 (p[4][0],p[4][1]),
                 (p[10][0],p[10][1]),
                 (p[3][0],p[3][1]),
                 (p[5][0],p[5][1]),
                 (p[2][0],p[2][1]),
                 (p[0][0],p[0][1]),
                 (p[11][0],p[11][1])),
                fill=(136,232,232),
                width=1
            )
            draw.ellipse([
            (p[1][0]-30.0,p[1][1]-30.0),
            (p[1][0]+30.0,p[1][1]+30.0)
            ],outline=(136,232,232),width=5)

        elif step==0:
            draw.line(
                ((p[11][0],p[ 11][1]),
                 (p[1][0],p[1][1]),
                 (p[2][0],p[2][1]),
                 (p[5][0],p[5][1]),
                 (p[4][0],p[4][1]),
                 (p[10][0],p[10][1]),
                 (p[3][0],p[3][1]),
                 (p[5][0],p[5][1]),
                 (p[2][0],p[2][1]),
                 (p[0][0],p[0][1]),
                 (p[11][0],p[11][1])),
                fill=(136,232,232),
                width=1
            )
        elif step==2:
            draw.line(
                ((p[6][0],p[6][1]),
                 (p[0][0],p[0][1]),
                 (p[12][0],p[12][1]),
                 (p[5][0],p[5][1]),
                 (p[13][0],p[13][1]),
                 (p[1][0],p[1][1]),
                 (p[7][0],p[7][1])),
                fill=(136,232,232),
                width=1
            )
            draw.line(
                ((p[11][0],p[11][1]),
                 (p[7][0],p[7][1]),
                 (p[9][0],p[9][1]),
                 (p[10][0],p[10][1]),
                 (p[8][0],p[8][1]),
                 (p[6][0],p[6][1]),
                 (p[11][0],p[11][1])),
                fill=(136,232,232),
                width=1
            )
            draw.line(
                ((p[11][0],p[11][1]),
                 (p[1][0],p[1][1]),
                 (p[2][0],p[2][1]),
                 (p[5][0],p[5][1]),
                 (p[4][0],p[4][1]),
                 (p[10][0],p[10][1]),
                 (p[3][0],p[3][1]),
                 (p[5][0],p[5][1]),
                 (p[2][0],p[2][1]),
                 (p[0][0],p[0][1]),
                 (p[11][0],p[11][1])),
                fill=(136,232,232),
                width=1
            )


        """draw.line(
                ((p[11][0],p[11][1]),(p[7][0],p[7][1]),
            (p[9][0],p[9][1]),
            (p[10][0],p[10][1]),(p[8][0],p[8][1]),
            (p[6][0],p[6][1]),
            (p[11][0],p[11][1])),
        fill=(136,232,232),width=1)

        draw.line(
                ((p[11][0],p[11][1]),(p[1][0],p[1][1]),
            (p[4][0],p[4][1]),
            (p[10][0],p[10][1]),(p[3][0],p[3][1]),
            (p[0][0],p[0][1]),
            (p[11][0],p[11][1])),
        fill='white',width=1)

        draw.line(
                ((p[6][0],p[6][1]),(p[0][0],p[0][1]),
            (p[1][0],p[1][1]),
            (p[7][0],p[7][1]),(p[4][0],p[4][1]),
            (p[5][0],p[5][1]),
            (p[3][0],p[3][1]),(p[6][0],p[6][1])),
        fill='white',width=1)

        draw.line(((p[0][0],p[0][1]),(p[2][0],p[2][1]),
            (p[4][0],p[4][1])),
        fill='white',width=1)

        draw.line(( (p[1][0],p[1][1]),
            (p[2][0],p[2][1]),
            (p[3][0],p[3][1])),
        fill='white',width=1)
        
        draw.line(( (p[8][0],p[8][1]),
            (p[3][0],p[3][1])),
        fill='white',width=1)

        draw.line(( (p[4][0],p[4][1]),
            (p[9][0],p[9][1])),
        fill='white',width=1)

        draw.line(( (p[8][0],p[8][1]),
            (p[0][0],p[0][1])),
        fill='white',width=1)

        draw.line(( (p[1][0],p[1][1]),
            (p[9][0],p[9][1])),
        fill='white',width=1)
        """

        return img_copy
