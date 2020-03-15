import cv2


def show_feed(frame, window_name="Feed"):
    """
    show_feed for Video Stream
    
    The Function takes a frame and an optional stream name.
    It displays the frame and waits 1 ms for a keypress. 
    If the user presses 'q', the function returns True.
    If the user presses space, the video stream pauses, and it continues on any keypress.
    The function returns False by default
    
    Args:
        frame (Numpy Array): Video Feed Frame
        window_name (str, optional): Window name for the feed. Defaults to "Feed".
    
    Returns:
        boolean: True for 'q' else False
    """

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)
    if key == ord(' '):
        cv2.waitKey(0)
        return False
    elif key == ord('q'):
        return True
    else: 
        return False


def fix_boxes(x1, y1, x2, y2, frame_w, frame_h):
    """
    fix_boxes for the model predictions
    
    This function removes negative coordinate outputs of the object detector and replaces them with 0. 
    It also limits the maximum coordinate outputs of the object detector to the frame width and height.
    This avoids slicing and other errors
    
    Args:
        x1 (int): xmin or the starting of the x axis
        y1 (int): ymin or the starting of the y axis
        x2 (int): xmax or the ending of the x axis
        y2 (int): ymax or the ending of the y axis
        frame_w (int): max width of the frame
        frame_h (int): max height of the frame
    
    Returns:
        tuple: x1, y1, x2, y2
    """
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    return x1, y1, x2, y2