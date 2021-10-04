from xml.dom import minidom
import os


def write_voc(output_folder, annos):

    for anno_dict in annos:
        impl = minidom.getDOMImplementation()
        dom = impl.createDocument(None, None, None)  # namespaceURI,qualifiedName,doctype

        # 创建根节点
        root = dom.createElement('annotation')

        # 创建filename节点
        filename_node = dom.createElement('filename')

        filename = anno_dict['filename']

        filename_node_txt = dom.createTextNode(filename)
        filename_node.appendChild(filename_node_txt)
        root.appendChild(filename_node)

        # 创建size节点
        size = anno_dict['size']
        size_node = dom.createElement('size')
        width_node = dom.createElement('width')
        width_node_txt = dom.createTextNode(str(size['width']))
        width_node.appendChild(width_node_txt)
        height_node = dom.createElement('height')
        height_node_txt = dom.createTextNode(str(size['height']))
        height_node.appendChild(height_node_txt)

        depth_node = dom.createElement('depth')
        depth_node_txt = dom.createTextNode(str(size['depth']))
        depth_node.appendChild(depth_node_txt)

        size_node.appendChild(width_node)
        size_node.appendChild(height_node)
        size_node.appendChild(depth_node)
        root.appendChild(size_node)


        raw_shp_node = dom.createElement('raw_shp')
        raw_shp_node_txt = dom.createTextNode(anno_dict['raw_shp'])
        raw_shp_node.appendChild(raw_shp_node_txt)
        root.appendChild(raw_shp_node)

        raw_tif_node = dom.createElement('raw_tif')
        raw_tif_node_txt = dom.createTextNode(anno_dict['raw_tif'])
        raw_tif_node.appendChild(raw_tif_node_txt)
        root.appendChild(raw_tif_node)


        sample_shfit_x, sample_shfit_y = anno_dict['shift']
        shift_node = dom.createElement('shift')
        shift_x_node = dom.createElement('shift_x')
        shift_x_node_txt = dom.createTextNode(str(sample_shfit_x))
        shift_x_node.appendChild(shift_x_node_txt)

        shift_y_node = dom.createElement('shift_y')
        shift_y_node_txt = dom.createTextNode(str(sample_shfit_y))
        shift_y_node.appendChild(shift_y_node_txt)

        shift_node.appendChild(shift_x_node)
        shift_node.appendChild(shift_y_node)

        root.appendChild(shift_node)







        # 创建segmented节点
        segmented_node = dom.createElement('segmented')
        segmented_node_txt = dom.createTextNode(str(0))
        segmented_node.appendChild(segmented_node_txt)
        root.appendChild(segmented_node)

        # 创建object节点
        samples = anno_dict['samples']
        for sample in samples:
            object_node = dom.createElement('object')

            name_node = dom.createElement('name')
            name_node_txt = dom.createTextNode(sample['name'])
            name_node.appendChild(name_node_txt)
            bndbox_node = dom.createElement('bndbox')

            xmin, ymin, xmax, ymax = sample['bbox']

            xmin_node = dom.createElement('xmin')
            xmin_node_txt = dom.createTextNode(str(xmin))
            xmin_node.appendChild(xmin_node_txt)
            ymin_node = dom.createElement('ymin')
            ymin_node_txt = dom.createTextNode(str(ymin))
            ymin_node.appendChild(ymin_node_txt)
            xmax_node = dom.createElement('xmax')
            xmax_node_txt = dom.createTextNode(str(xmax))
            xmax_node.appendChild(xmax_node_txt)
            ymax_node = dom.createElement('ymax')
            ymax_node_txt = dom.createTextNode(str(ymax))
            ymax_node.appendChild(ymax_node_txt)

            bndbox_node.appendChild(xmin_node)
            bndbox_node.appendChild(ymin_node)
            bndbox_node.appendChild(xmax_node)
            bndbox_node.appendChild(ymax_node)

            object_node.appendChild(name_node)
            object_node.appendChild(bndbox_node)
            root.appendChild(object_node)


        dom.appendChild(root)
        xml_filename = filename.split('.')[0] + '.xml'
        output_fullpath = os.path.join(output_folder, xml_filename)
        f = open(output_fullpath, 'w', encoding='utf-8')
        dom.writexml(f,  addindent='    ', newl='\n', encoding='utf-8')
        f.close()