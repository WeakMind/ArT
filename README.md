# ArT
Arbitrary Shaped Text Detection from a given image. Unlike normal text detection, in this project the aim was to detect English/Latin language text of any arbitrary shape. No restriction of it being rectangular in shape.

Dataset - http://rrc.cvc.uab.es/?ch=14

Language Used - Python

Framework Used - Pytorch

Architecture - Instance segmentation approach is used to build this model. Text instances are first segmented out by linking pixels within the same instance together. Text bounding boxes are then extracted directly from the segmentation result without location regression. using this approach, it doesnt matter what the shape of the text is, it will segment and eventually bound the text . Bounding boxes of one connected component are extracted through minAreaRect in OpenCV. U-Net model architecture is used as segmentation choice.


