import os, shutil

print("ssss")
train_folder = r'/home/dl/Documents/ssd.pytorch-master/data/VOCdevkit_shandong_oilsite/COCO/train'
test_folder = r'/home/dl/Documents/ssd.pytorch-master/data/VOCdevkit_shandong_oilsite/COCO/test'

txt_folder = r'/home/dl/Documents/ssd.pytorch-master/data/VOCdevkit_shandong_oilsite/VOC2007/ImageSets/Main'
train_filename = r'trainval.txt'
test_filename = r'test.txt'

train_fullpath = os.path.join(txt_folder, train_filename)
test_fullpath = os.path.join(txt_folder, test_filename)

xml_folder = r'/home/dl/Documents/ssd.pytorch-master/data/VOCdevkit_shandong_oilsite/VOC2007/Annotations'
img_folder = r'/home/dl/Documents/ssd.pytorch-master/data/VOCdevkit_shandong_oilsite/VOC2007/JPEGImages'

print(train_fullpath)

with open(train_fullpath, 'r') as train_f:
    lines = train_f.readlines()
    for line in lines:
        line = line.strip()
        xml_filename = line + '.xml'
        img_filename = line + '.jpg'
        xml_fullpath = os.path.join(xml_folder, xml_filename)
        img_fullpath = os.path.join(img_folder, img_filename)
        
        shutil.copy(xml_fullpath, os.path.join(train_folder, 'xmls', xml_filename))
        shutil.copy(img_fullpath, os.path.join(train_folder, 'imgs', img_filename))



with open(test_fullpath, 'r') as test_f:
    lines = test_f.readlines()
    for line in lines:
        line = line.strip()
        xml_filename = line + '.xml'
        img_filename = line + '.jpg'
        xml_fullpath = os.path.join(xml_folder, xml_filename)
        img_fullpath = os.path.join(img_folder, img_filename)
        
        shutil.copy(xml_fullpath, os.path.join(test_folder, 'xmls', xml_filename))
        shutil.copy(img_fullpath, os.path.join(test_folder, 'imgs', img_filename))        