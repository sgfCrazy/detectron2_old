import json


def write_coco(josn_fullpath, annos):

    json_str = {}
    json_str["info"] = {
        "description": "",
        "url": "",
        "year": 2021
    }
    image_id = 1
    annotation_id = 1
    category_id = 1
    images = []
    annotations = []
    categories = []


    for anno in annos:
        image_item = {}

        img_filename = anno['filename']
        size = anno['size']
        width = int(size['width'])
        height = int(size['height'])

        image_item['file_name'] = img_filename
        image_item['height'] = height
        image_item['width'] = width
        image_item['id'] = image_id

        samples = anno['samples']

        for sample in samples:
            annotation_item = {}

            name = sample['name']
            category_id_sample = None
            for category in categories:
                if name == category['name']:
                    category_id_sample = category['id']
                    break

            if category_id_sample is None:
                category_id_sample = category_id
                category = {
                    "supercategory" : "none",
                    "id": category_id_sample,
                    "name": name
                }
                categories.append(category)
                category_id += 1


            xmin, ymin, xmax, ymax = [int(c) for c in sample['bbox']]
            w, h = xmax - xmin, ymax - ymin
            annotation_item['image_id'] = image_id
            annotation_item['bbox'] = [xmin, ymin, w, h]
            annotation_item['category_id'] = category_id_sample
            annotation_item['id'] = annotation_id
            annotation_item['iscrowd'] = 0
            annotation_item['segmentation'] = []
            annotation_item['area'] = w * h
            annotations.append(annotation_item)
            annotation_id += 1
        images.append(image_item)
        image_id += 1

        json_str["images"] = images
        json_str["annotations"] = annotations
        json_str["categories"] = categories

        with open(josn_fullpath, 'w') as f:
            json.dump(json_str, f)