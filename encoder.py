import cv2
import numpy as np
import face_recognition
import os
import pickle
from imutils import paths

class image_encoder:
        
    #Method to read images with its corresponding name
    def name_img_reader(self,path_to_images):
        #Get image path
        self.image_paths = list(paths.list_images(path_to_images))
        
        self.images = []
        self.class_names = []
        
        #loop through the images and grab each image and its corresponding classname
        for indx,image in enumerate(self.image_paths):
            self.name = image.split(os.path.sep)[-2]
            self.image = cv2.imread(image)

            self.images.append(self.image)
            self.class_names.append(self.name)
            
        return self.class_names,self.images


    def encode_images(self,args):
        """
        path_to_images: The path to the image directory
        """

        self.known_encoding = []
        self.known_names = []
        
        #Calling the name_ima_reader in the same class
        class_names,images = self.name_img_reader(args.path_to_images)
        
        #zip into tuples
        for name,img in zip(class_names,images):
            #to rgb color
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #get the face of the image
            face_extract = face_recognition.face_locations(rgb_img,model='hog')
            encoded_img = face_recognition.face_encodings(rgb_img,face_extract)
            # loop over the encodings
            for encoding in encoded_img:
                self.known_encoding.append(encoding)
                self.known_names.append(name)

        self.known_encoded_dict = {'names':self.known_names,'images':self.known_encoding}

        f = open('encoded_images','wb')
        f.write(pickle.dumps(self.known_encoded_dict))
        f.close()
        print('Learning Completed')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Commands for facial recognition')
    parser.add_argument('--path_to_images',default='known_images_folder',type=str)
    args = parser.parse_args()
    
    encoder = image_encoder()

    encoder.encode_images(args)

