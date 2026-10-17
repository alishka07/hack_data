"""Portable NumPy equivalent of cython_bbox's inclusive-pixel overlap convention."""
import numpy as np

def bbox_overlaps(boxes, query_boxes):
    boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    other=np.asarray(query_boxes,dtype=np.float64).reshape(-1,4)
    left=np.maximum(boxes[:,None,:2],other[None,:,:2])
    right=np.minimum(boxes[:,None,2:],other[None,:,2:])
    intersection=np.maximum(0,right-left+1).prod(axis=-1)
    first=np.maximum(0,boxes[:,2:]-boxes[:,:2]+1).prod(axis=-1)
    second=np.maximum(0,other[:,2:]-other[:,:2]+1).prod(axis=-1)
    union=first[:,None]+second[None,:]-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
