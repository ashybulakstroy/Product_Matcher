"""Synthetic reproducible fixture. This is NOT evidence on real Torgsoft photos."""
from __future__ import annotations
import cv2
import numpy as np


def receipt_grid(width=840,height=1060):
    white=np.full((height,width,3),244,np.uint8)
    xs=[120,183,525,628,742]
    ys=[150,206,298,389,488,596,702,780]
    for x in xs:cv2.line(white,(x,ys[0]),(x,ys[-1]),(90,90,90),2)
    for y in ys:cv2.line(white,(xs[0],y),(xs[-1],y),(95,95,95),2)
    for i,y in enumerate(ys[:-1]):
        cv2.putText(white,str(i+1),(135,y+27),cv2.FONT_HERSHEY_SIMPLEX,.7,(60,60,60),2)
        cv2.putText(white,'TOVAR',(206,y+27),cv2.FONT_HERSHEY_SIMPLEX,.7,(70,70,70),2)
    return white,xs,ys


def deform(img,mode='wave'):
    h,w=img.shape[:2]
    gy,gx=np.indices((h,w),dtype=np.float32)
    if mode=='wave':
        mx=gx+7*np.sin(gy/100.)+3*np.sin(gx/65.)
        my=gy+9*np.sin(gx/120.)+3*np.sin(gy/110.)
    elif mode=='tilt':
        mx=gx+(gy-h/2)*.045
        my=gy+gx*.028
    else:raise ValueError(mode)
    return cv2.remap(img,mx.astype(np.float32),my.astype(np.float32),
                     cv2.INTER_LINEAR,borderValue=(255,255,255))
