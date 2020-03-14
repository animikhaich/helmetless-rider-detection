import cv2, imutils, os, shutil, tqdm, glob, random
from xml.etree.ElementTree import Element, SubElement, tostring
from PIL import Image, ImageDraw
import numpy as np
import lxml.etree as etree

# TODO: Add if __name__=='__main__' condition and make the script runnable
# TODO: Test on a random Dataset

class DataGenerator:
    
    def __init__(
        self,
        bg_root_path='backgrounds',
        objects_root_paths='objects',
        object_to_bg_width_ratio=0.1,
        min_objects=1,
        max_objects=5,
        ):

        # Get Backgrounds
        self.all_background_paths = glob.glob(os.path.join(bg_root_path, '*'))

        # Verify if background images are provided
        if len(self.all_background_paths) < 1:
            raise Exception(f'No background images provided. Please provide background images in the folder named: {bg_root_path}')

        # Get Objects
        self.all_objects_paths = glob.glob(os.path.join(objects_root_paths, '*', '*'))

        # Verify if objects are provided
        if len(self.all_objects_paths) < 1:
            raise Exception(f'No Objects provided. Please provide object images in the folder named: {objects_root_paths}')

        # Declare Variables
        self.output_image_path = os.path.join('outputs', 'images')
        self.output_annot_path = os.path.join('outputs', 'annotations')
        self.object_to_bg_width_ratio = object_to_bg_width_ratio
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.existing_coords = []

        # Verify and create the output directories
        if not os.path.isdir(self.output_image_path): os.makedirs(self.output_image_path)
        if not os.path.isdir(self.output_annot_path): os.makedirs(self.output_annot_path)

    def generate_coordinates(self, bg_w, bg_h, object_w, object_h, max_tries=100):
        for _ in range(max_tries):
            # Randomly generated X and Y coordinates
            x_coord = np.random.randint(0, bg_w-object_w)
            y_coord = np.random.randint(0, bg_h-object_h)
            
            # Extract the coordinates of the image
            x1, y1, x2, y2 = x_coord, y_coord, x_coord + object_w, y_coord + object_h
            
            # Initialize Overlaps, Because default value is necessary
            overlaps = [False]

            # Check if any of the existing coordinates overlap with the randomly created one
            for x3, y3, x4, y4 in self.existing_coords:
                diff_width = min(x2, x4) - max(x1, x3)
                diff_height = min(y2, y4) - max(y1, y3)

                intersection_area = diff_width * diff_height

                # Check if there is overlap by checking if the intersection area is negative
                if diff_width < 0 and diff_height < 0 or intersection_area < 0:
                    overlaps.append(False)
                else:
                    overlaps.append(True)
            
            # if there are no overlaps with any existing coordinates, return the new coordinates
            if not any(overlaps):
                self.existing_coords.append([x1, y1, x2, y2])
                return True, (x1, y1, x2, y2)

        # If No viable coordinates are found, return False
        return False, (0, 0, 0, 0)

    def main(self, total_combinations=200):
        for combination in tqdm.tqdm(range(total_combinations)):
            bg_image_path = np.random.choice(self.all_background_paths, 1)[0]
            object_paths = np.random.choice(self.all_objects_paths, np.random.randint(self.min_objects, self.max_objects+1))
            
            # XML Elements
            top = Element('annotation')
            filename = SubElement(top, 'filename')
            xml_size = SubElement(top, 'size')
            xml_width = SubElement(xml_size, 'width')
            xml_height = SubElement(xml_size, 'height')
            xml_depth = SubElement(xml_size, 'depth')

            bg_image = Image.open(bg_image_path, 'r')
            bg_image = bg_image.convert('RGB')
            bg_w, bg_h = bg_image.size

            xml_width.text = str(bg_w)
            xml_height.text = str(bg_h)
            xml_depth.text = str(3)
            
            # Reset the existing coordinates for each new image
            self.existing_coords = []

            for object_path in object_paths:
                # Read the object without any modifications to it's number of channels
                object_image = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
                
                # Skip the object if it is in grayscale
                try:
                    object_h, object_w, channels = object_image.shape
                except:
                    continue
                
                # Convert the image to RGB by conserving the number of channels
                if channels == 4:
                    object_image = cv2.cvtColor(object_image, cv2.COLOR_BGRA2RGBA)
                else:
                    object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)

                # Scale the object as per the background based on width or height conserving the original ratio
                if object_h < object_w:
                    target_height = round(bg_h * self.object_to_bg_width_ratio)
                    object_image = imutils.resize(object_image, height=target_height)
                else:
                    target_width = round(bg_w * self.object_to_bg_width_ratio)
                    object_image = imutils.resize(object_image, width=target_width)

                # Get the width and height of the object and convert to PIL image
                object_h, object_w, channels = object_image.shape
                if channels == 4:
                    object_image = Image.fromarray(object_image, 'RGBA')
                else:
                    object_image = Image.fromarray(object_image)

                # Generate Random Coordinates
                generated, coordinates = self.generate_coordinates(bg_w, bg_h, object_w, object_h, max_tries=100)
                
                # if non overlapping coordinates aren't generated in 100 tries, continue to the next object
                if not generated: continue

                # Unpack the coordinates
                x1, y1, x2, y2 = coordinates

                # If the object is RGBA, paste as a mask on the background
                if channels == 4:
                    bg_image.paste(object_image, (x1, y1), mask=object_image)
                else:
                    bg_image.paste(object_image, (x1, y1))

                # Write XML Elements
                xml_object = SubElement(top, 'object')
                xml_object_name = SubElement(xml_object, 'name')
                xml_object_bndbox = SubElement(xml_object, 'bndbox')
                xml_object_bndbox_xmin = SubElement(xml_object_bndbox, 'xmin')
                xml_object_bndbox_xmax = SubElement(xml_object_bndbox, 'xmax')
                xml_object_bndbox_ymin = SubElement(xml_object_bndbox, 'ymin')
                xml_object_bndbox_ymax = SubElement(xml_object_bndbox, 'ymax')

                xml_object_name.text = os.path.split(object_path)[0].split('/')[-1]
                xml_object_bndbox_xmin.text = str(x1)
                xml_object_bndbox_ymin.text = str(y1)
                xml_object_bndbox_xmax.text = str(x2)
                xml_object_bndbox_ymax.text = str(y2)

            # Save the synthetic image
            bg_image.save(os.path.join(self.output_image_path, f"{combination}.jpg"))
            filename.text = f"{combination}.jpg"

            # Write the XML File
            xml_str = tostring(top).decode()
            etree_xml = etree.fromstring(xml_str)
            with open(os.path.join(self.output_annot_path, f"{combination}.xml"), 'w') as f:
                f.write(etree.tostring(etree_xml, pretty_print=True).decode())
            
