# Face Aligning and Cropping using MTCNN
__Input__: a image and a list of face rects in this image;  
__Output__: a list of aligned and cropped face chips.  
## requirements
```
pycaffe
numpy
cv2
```

## how-to  
refer to (face_aligner.py)[./face_aligner.py]

1. init a face aligner:
```python
from face_aligner import FaceAligner
caffe_model_path = '../model'
aligner = FaceAligner(caffe_model_path)
```

2. input a image and face rects list:
```python
img_path = '../test_imgs/Marilyn_Monroe_0002.jpg'
    face_rect1 = [
            [
                91,
                57
            ],
            [
                173,
                57
            ],
            [
                173,
                180
            ],
            [
                91,
                180
            ]
        ]

    face_rects = [face_rect1]
```

3. get a list of aligned face crops:
```python
face_chips = aligner.get_face_chips(img,face_rects)
```

4. (optional) show the face chips with cv2:
```python
    for i, chip in enumerate(face_chips):

        save_name = osp.join(save_dir, name + '_chip' + str(i) + ext)
        cv2.imwrite(save_name, chip)

        if show_img:
            cv2.imshow('face_chip', chip)

            cv2.waitKey(0)
```
