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