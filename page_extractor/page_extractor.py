#! /usr/bin/env python3

__version__ = '1.0'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
import numpy as np
from sys import getsizeof
import random
from tqdm import tqdm
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

import gc
from tensorflow.python.keras import backend as K
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import xml.etree.ElementTree as ET
import warnings
import click
import time
import imutils
import xml.etree.ElementTree as ET
import json

warnings.filterwarnings('ignore')
#
__doc__ = \
    """
    tool to extract table form data from alto xml data
    """
class page_extractor:
    def __init__(self,dir_out, dir_models , image_dir = False, co_image = False, out_co = False, directory_images = False, out_page_bin = False, out_page_scaled = False, co_out_page_scaled = False, out_page_scaled_bin = False, dir_xmls = False, out_xmls = False, write_num_columns = False, columns_widths = False):
        self.image_dir = image_dir  # XXX This does not seem to be a directory as the name suggests, but a file
        self.dir_out = dir_out
        self.kernel = np.ones((5, 5), np.uint8)
        
        
        self.model_dir_of_binarization = dir_models + "/eynollah-binarization_20210425"
        self.model_dir_of_col_classifier = dir_models + "/eynollah-column-classifier_20210425"
        self.model_page_dir = dir_models + "/eynollah-page-extraction_20210425"
        #self.model_page_dir = dir_models
        self.co_image = co_image
        self.out_co = out_co
        self.out_page_bin = out_page_bin
        self.out_page_scaled = out_page_scaled
        self.co_out_page_scaled = co_out_page_scaled
        self.out_page_scaled_bin = out_page_scaled_bin
        #self.out_page_xmls = out_page_xmls
        self.dir_xmls = dir_xmls
        self.out_xmls = out_xmls
        self.write_num_columns = write_num_columns
        self.columns_widths = columns_widths
        
        self.dir_in = directory_images
        if self.dir_in:
            self.ls_imgs  = os.listdir(self.dir_in)
            self.model_page = self.our_load_model(self.model_page_dir)
            self.model_classifier = self.our_load_model(self.model_dir_of_col_classifier)
            self.model_bin = self.our_load_model(self.model_dir_of_binarization)
            
    def our_load_model(self, model_file):
        
        try:
            model = load_model(model_file, compile=False)
        except:
            model = load_model(model_file , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})

        return model
        
    def get_image_and_co_image(self,image_dir,co_image,out_co,dir_out,image_name):
        
        file_stem = image_name.split('.')[0]
        if self.dir_in:
            image_dir = os.path.join(image_dir,image_name)
        else:
            pass
        if co_image:
            co_image = os.path.join(co_image,file_stem+'.png')
            #out_co_to_wr = os.path.join(out_co,file_stem+'.png')
    
        #dir_out_to_wr = os.path.join(dir_out,image_name)
        
        self.image = cv2.imread(image_dir)
        if co_image:
            #print(co_image,'co_image')
            self.image_co = cv2.imread(co_image)
        #if co_image:
            #return dir_out_to_wr, out_co_to_wr
        #else:
            #return dir_out_to_wr
        
    def read_image(self):
        self.image = cv2.imread(self.image_dir)
        if self.co_image:
            self.image_co = cv2.imread(self.co_image)
        
            
    def resize_image(self, img_in, input_height, input_width):
        return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

            
    def start_new_session_and_model(self, model_dir):
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True

        #session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model
    
    def do_prediction(self,patches,img,model,marginal_of_patch_percent=0.1):
        
        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        n_classes = model.layers[len(model.layers) - 1].output_shape[3]
        


        if patches:
            if img.shape[0]<img_height_model:
                img=self.resize_image(img,img_height_model,img.shape[1])
                
            if img.shape[1]<img_width_model:
                img=self.resize_image(img,img.shape[0],img_width_model)
            
            #print(img_height_model,img_width_model)
            #margin = int(0.2 * img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin


            img = img / float(255.0)
            #print(sys.getsizeof(img))
            #print(np.max(img))
            
            img=img.astype(np.float16)
            
            #print(sys.getsizeof(img))

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                        
                    

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                    label_p_pred = model.predict(
                        img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]), verbose=0)

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i==0 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)
            del img
            del mask_true
            del seg_color
            del seg
            del img_patch
                
        if not patches:
            img_h_page=img.shape[0]
            img_w_page=img.shape[1]
            img = img /float( 255.0)
            img = self.resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(
                img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), verbose=0)

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color =np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = self.resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
            
            
            del img
            del seg_color
            del label_p_pred
            del seg
        del model
        gc.collect()
        
        return prediction_true
    
    def crop_image_inside_box(self, box, img_org_copy):
        image_box = img_org_copy[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]
            
    def extract_page(self):
        patches=False
        if not self.dir_in:
            model_page = self.start_new_session_and_model(self.model_page_dir)
        ###img = self.otsu_copy(self.image)
        for ii in range(1):
            img = cv2.GaussianBlur(self.image, (5, 5), 0)

        if not self.dir_in:
            img_page_prediction=self.do_prediction(patches,img,model_page)
        else:
            img_page_prediction=self.do_prediction(patches,img,self.model_page)
        
        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)
        

        if x<=30:
            w=w+x
            x=0
        if (self.image.shape[1]-(x+w) )<=30:
            w=w+(self.image.shape[1]-(x+w) )
        
        if y<=30:
            h=h+y
            y=0
        if (self.image.shape[0]-(y+h) )<=30:
            h=h+(self.image.shape[0]-(y+h) )
            
            

        box = [x, y, w, h]

        croped_page, page_coord = self.crop_image_inside_box(box, self.image)
        
        if self.co_image:
            co_image_page = self.image_co[page_coord[0]:page_coord[1] , page_coord[2]:page_coord[3] , :]
        
        else:
            co_image_page = None
        
        self.cont_page=[]
        self.cont_page.append( np.array( [ [ page_coord[2] , page_coord[0] ] , 
                                                    [ page_coord[3] , page_coord[0] ] ,
                                                    [ page_coord[3] , page_coord[1] ] ,
                                                [ page_coord[2] , page_coord[1] ]] ) )


        del contours
        del thresh
        del img
        del imgray

        return croped_page, page_coord, co_image_page
    
    def number_of_columns(self,image_page,page_coord, file_name):
        
        if not self.dir_in:
            model_num_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)
            img_1ch = cv2.imread(self.image_dir, 0)
        else:
            img_1ch = cv2.imread(os.path.join(self.dir_in,file_name), 0)
        width_early = img_1ch.shape[1]
        img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

        img_1ch = img_1ch / 255.0

        img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)

        img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
        img_in[0, :, :, 0] = img_1ch[:, :]
        img_in[0, :, :, 1] = img_1ch[:, :]
        img_in[0, :, :, 2] = img_1ch[:, :]
        
        if not self.dir_in:
            label_p_pred = model_num_classifier.predict(img_in, verbose=0)
        else:
            label_p_pred = self.model_classifier.predict(img_in, verbose=0)
        #if not self.dir_in:
            #label_p_pred = model_num_classifier.predict(img_in, verbose=0)
        #else:
            #label_p_pred = self.model_classifier.predict(img_in, verbose=0)

        num_col = np.argmax(label_p_pred[0]) + 1
        
        return num_col
    def calculate_width_height_by_columns(self, img, num_col, ls_widths):
        img_w_new = ls_widths[str(num_col)]
        img_h_new = int(img.shape[0] / float(img.shape[1]) * img_w_new)
        #img_w_new = int( num_col*500 +300 )
        #img_h_new = int(img.shape[0] / float(img.shape[1]) * img_w_new)
        return img_w_new, img_h_new
    
    def do_binarization(self,img):
        
        if not self.dir_in:
            model_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
            prediction_bin = self.do_prediction(True, img, model_bin)
        else:
            prediction_bin = self.do_prediction(True, img, self.model_bin)
        
        prediction_bin=prediction_bin[:,:,0]
        prediction_bin = (prediction_bin[:,:]==0)*1
        prediction_bin = prediction_bin*255
        
        prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

        prediction_bin = prediction_bin.astype(np.uint8)
        
        return prediction_bin
    def run(self):
        #self.dir_in = False
        
        if not self.dir_in:
            self.ls_imgs = [1]
            
        dir_page_images_to_write= '/home/vahid/Documents/main_regions_new_concept_training_dataset/training_data_asiatca_sbb_new_concept/images_page'
        dir_page_label = '/home/vahid/Documents/main_regions_new_concept_training_dataset/training_data_asiatca_sbb_new_concept/labels_page'
        
        dir_page_images_bin_to_write = '/home/vahid/Documents/main_regions_new_concept_training_dataset/training_data_asiatca_sbb_new_concept/images_page_bin'
        
        dir_page_images_scaled_to_write= '/home/vahid/Documents/main_regions_new_concept_training_dataset/training_data_asiatca_sbb_new_concept/images_page_scaled'
        
        for img_name in tqdm(self.ls_imgs):
            
            
            if self.dir_in:
                file_stem = img_name.split('.')[0]
                dir_in = self.dir_in
            else:
                img_name = self.image_dir.split('/')[-1]
                file_stem = img_name.split('.')[0]
                dir_in = self.image_dir
                

            self.get_image_and_co_image(dir_in,self.co_image,self.out_co, self.dir_out, img_name)
            
                
        
            image_page,page_coord, co_image_page=self.extract_page()
            if self.dir_out:
                cv2.imwrite(os.path.join(self.dir_out,img_name),image_page)
                
            if self.co_image:
                cv2.imwrite(os.path.join(self.out_co,file_stem+'.png'),co_image_page)
            
            if self.out_page_bin or self.out_page_scaled_bin:
                img_bin = self.do_binarization(image_page)
            if self.out_page_bin:
                cv2.imwrite(os.path.join(self.out_page_bin,img_name),img_bin)
                
            if self.out_page_scaled or self.out_page_scaled_bin:
                with open(self.columns_widths) as f:
                    widths_dict = json.load(f)
                ls_widths = widths_dict['widths']
                num_col = self.number_of_columns(image_page,page_coord,img_name)
                
                img_w_new, img_h_new = self.calculate_width_height_by_columns(image_page, num_col, ls_widths)
                
                #print(num_col, img_w_new, img_h_new)
            if self.write_num_columns and not self.out_page_scaled and not self.out_page_scaled:
                num_col = self.number_of_columns(image_page,page_coord,img_name)
                #print(num_col)
            if self.out_page_scaled:
                img_page_resize = self.resize_image(image_page,img_h_new,img_w_new)
                cv2.imwrite(os.path.join(self.out_page_scaled,img_name),img_page_resize)
                
            if self.out_page_scaled_bin:
                img_bin_resized  = self.resize_image(img_bin,img_h_new,img_w_new)
                cv2.imwrite(os.path.join(self.out_page_scaled_bin,img_name),img_bin_resized)
                
                
                
                
            if self.co_out_page_scaled:
                label_resized  = self.resize_image(co_image_page,img_h_new,img_w_new)
                cv2.imwrite(os.path.join(self.co_out_page_scaled,file_stem+'.png'),label_resized)
                
            if self.dir_xmls:
                parser = ET.XMLParser(encoding="utf-8")
                #tree = ET.parse(os.path.join(self.dir_xmls,file_stem+'.xml'))
                tree = ET.parse(os.path.join(self.dir_xmls,file_stem+'.xml'), parser = ET.XMLParser(encoding = 'iso-8859-5'))
                root=tree.getroot()
                alltags=[elem.tag for elem in root.iter()]
                link=alltags[0].split('}')[0]+'}'
                
                name_space = alltags[0].split('}')[0]
                name_space = name_space.split('{')[1]
                
                page_element = root.find(link+'Page')
                metadata_element = root.find(link+'Metadata')
                
                
                type_of_coords_point = False
                
                if (link+'Point' in alltags):
                    type_of_coords_point = True

                        
                if self.write_num_columns:
                    ##subelement_col = ET.SubElement(metadata_element,'Columns_Num')
                    ##subelement_col.text = str(num_col)
                    ##metadata_element.set('Columns_num',str(num_col))
                    
                    comments_is_a_submetadata = False
                    for child2 in metadata_element:
                        tag2 = child2.tag
                        if tag2.endswith('}Comments') or tag2.endswith('}comments'):
                            comments_is_a_submetadata = True
                            text_comments = child2.text
                            if text_comments:
                                text_comments = text_comments +'num_col'+str(num_col)+'num_col'
                            else:
                                text_comments = 'num_col'+str(num_col)+'num_col'
                            child2.text = text_comments
                    if not comments_is_a_submetadata:
                        subelement_col = ET.SubElement(metadata_element,'Comments')
                        subelement_col.text = 'num_col'+str(num_col)+'num_col'
                
                
                if (link+'PrintSpace' in alltags) or  (link+'Border' in alltags):
                    pass
                else:
                    printspace_subelement = ET.Element('PrintSpace')
                        
                    
            
                    #page_print_sub=ET.SubElement(page, 'Border')
                    coord_page = ET.SubElement(printspace_subelement, 'Coords')
                    
                    if not type_of_coords_point:
                        points_page_print=''
                        
                        self.scale_x = 1
                        self.scale_y = 1

                        for lmm in range(len(self.cont_page[0])):
                            if len(self.cont_page[0][lmm])==2:
                                points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][0])/self.scale_x ) )
                                points_page_print=points_page_print+','
                                points_page_print=points_page_print+str( int( (self.cont_page[0][lmm][1])/self.scale_y ) )
                            else:
                                points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][0])/self.scale_x) )
                                points_page_print=points_page_print+','
                                points_page_print=points_page_print+str( int((self.cont_page[0][lmm][0][1])/self.scale_y) )

                            if lmm<(len(self.cont_page[0])-1):
                                points_page_print=points_page_print+' '
                        coord_page.set('points',points_page_print)
                    else:
                        for lmm in range(len(self.cont_page[0])):
                            if len(self.cont_page[0][lmm])==2:
                                coord_page_point_sub_element = ET.SubElement(coord_page,'Point')
                                coord_page_point_sub_element.set('x', str( int( (self.cont_page[0][lmm][0]) )) )
                                coord_page_point_sub_element.set('y', str( int( (self.cont_page[0][lmm][1]) )) )
                            else:
                                coord_page_point_sub_element = ET.SubElement(coord_page,'Point')
                                coord_page_point_sub_element.set('x', str( int( (self.cont_page[0][lmm][0][0]) )) )
                                coord_page_point_sub_element.set('y', str( int( (self.cont_page[0][lmm][0][1]) )) )
                            
                                                             
                    
                    page_element.insert(0, printspace_subelement)
                
                
                ET.register_namespace("",name_space)
                tree.write(os.path.join(self.out_xmls, file_stem+'.xml'),xml_declaration=True,method='xml',encoding="utf8",default_namespace=None)
    
        


        

@click.command()
@click.option('--image', '-i', help='image filename')
@click.option('--directory_images', '-di', help='image filename')
@click.option('--out', '-o', help='output cropped page image will be written here.')
@click.option('--model', '-m', help='directory of model')
@click.option('--co_image', '-ci', help='corresponding image file name that will be cropped as main image. In the case that you dont have any co image this option is not needed.')
@click.option('--out_co', '-co', help='output name for corresponding image name with directory.  In the case that you dont have any co image this option is not needed.')
@click.option('--out_page_bin', '-opb', help='if given the image page will be binarized and the output will be written here.')
@click.option('--out_page_scaled', '-ops', help='if given the image page will be scaled with column classifier model and scaled page will be written here.')
@click.option('--out_page_scaled_bin', '-opsb', help='if given the image page will be binarized and scaled with column classifier model and output will be written here.')
@click.option('--co_out_page_scaled', '-cops', help='if given corresponding image file name will also be cropped and scaled and written here.')
#@click.option('--out_page_xmls', '-opx', help='if given extracted page will be written here as a new xml file with the same file name as they are in xml dir.')
@click.option('--dir_xmls', '-dx', help='dir of xml files.')
@click.option('--out_xmls', '-ox', help='output directory where modified xmls will be written.')
@click.option(
    "--write_num_columns",
    "-wnc",
    is_flag=True,
    help="if this parameter set to true, number of columns will be encoded in metadata comments subelement as num_col{#num_col}num_col",
)

@click.option(
    "--columns_widths",
    "-cws",
    help="json dictionary file where the width for each number of columns is given and scaling will be adjusted with those inputs.",
    type=click.Path(exists=True, dir_okay=False),
)

def main(out, model, image,co_image, out_co, directory_images, out_page_bin, out_page_scaled, co_out_page_scaled, out_page_scaled_bin, dir_xmls, out_xmls, write_num_columns, columns_widths):
    if (out_page_scaled or co_out_page_scaled or out_page_scaled_bin ) and not columns_widths:
        print("Error. You have activated one of scaling output directories but you have not provided columns_width json file.")
        sys.exit()
    x = page_extractor( out, model, image, co_image, out_co, directory_images, out_page_bin, out_page_scaled, co_out_page_scaled, out_page_scaled_bin, dir_xmls, out_xmls, write_num_columns, columns_widths)
    x.run()


if __name__ == "__main__":
    main()
  
