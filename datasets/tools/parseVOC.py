from xml.dom import minidom
import os


def parse_voc(xmls_folder):
    xmls_filename = os.listdir(xmls_folder)

    annos = []

    for xml_filename in xmls_filename:

        anno_dict = {}

        xml_fullpath = os.path.join(xmls_folder, xml_filename)

        dom = minidom.parse(xml_fullpath)
        root = dom.documentElement

        img_filename = root.getElementsByTagName('filename')[0].childNodes[0].data
        anno_dict['filename'] = img_filename

        size = root.getElementsByTagName('size')[0]

        width = size.getElementsByTagName('width')[0].childNodes[0].data
        height = size.getElementsByTagName('height')[0].childNodes[0].data
        depth = size.getElementsByTagName('depth')[0].childNodes[0].data

        anno_dict['size'] = {'depth': depth, 'height': height, 'width': width}
        anno_dict['samples'] = []

        objects = root.getElementsByTagName('object')

        for obj in objects:
            name = obj.getElementsByTagName('name')[0].childNodes[0].data
            bndbox = obj.getElementsByTagName('bndbox')[0]

            xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
            ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].data

            anno_dict['samples'].append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})

        annos.append(anno_dict)

    return annos