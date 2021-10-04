# -*- coding: utf-8 -*-
import os
from osgeo import gdal
import osr
import numpy as np
import cv2


# gdal包用于处理栅格数据，ogr用于处理矢量数据
class MyRaster:

    def getSRSPair(self, filename):
        dataset = gdal.Open(filename)
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs

    # 读图像文件
    def readImg(self, filename):
        # 打开文件
        dataset = gdal.Open(filename)

        # 栅格矩阵的列数
        im_width = dataset.RasterXSize

        # 栅格矩阵的行数
        im_height = dataset.RasterYSize

        # GeoTransform[0],GeoTransform[3]  左上角位置
        # GeoTransform[1]是像元宽度
        # GeoTransform[5]是像元高度
        # 如果影像是指北的,GeoTransform[2]和GeoTransform[4]这两个参数的值为0。
        im_geotrans = dataset.GetGeoTransform()

        # 地图投影信息
        im_proj = dataset.GetProjection()

        # 将数据写成数组，对应栅格矩阵
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
        del dataset

        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def writeImg(self, filename, im_proj, im_geotrans, im_data):

        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        im_bands, im_height, im_width = im_data.shape if len(im_data.shape) == 3 else (1, *im_data.shape)

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    def splitTif(self, splitXY, tif_fullpath=None, output_folder=None, prefix='', fill_pixel=0, mode=1):
        # mode = 1 少的补全 mode = 2 少的不要 mode=3 少的不管
        splitX, splitY = splitXY
        os.makedirs(output_folder, exist_ok=True)
        # 读数据
        proj, geotrans, data = self.readImg(tif_fullpath)
        height, width = data.shape[-2:]
        # 如果是单波段，则添加一个维度，统一为3波段
        data = data if len(data.shape) == 3 else data[np.newaxis, :, :]

        pad_height = splitY - height % splitY
        pad_width = splitX - width % splitX
        if mode == 1:
            data = np.pad(data, ((0, 0), (pad_height, 0), (pad_width, 0)), 'constant', constant_values=fill_pixel)
        elif mode == 2:
            if width - width % splitX <= 0 or height - height % splitY <= 0:
                print('按此模式剪裁没了。。')
                return
            data = data[:, :height - height % splitY, :width - width % splitX]
        elif mode != 3:
            print(f"mode value: {mode} error! ")
            return

        channel, height, width = data.shape
        # 为了可以修改，将tuple改为list
        geotrans = list(geotrans)
        leftX, leftY = geotrans[0], geotrans[3]

        step_num_Y = height // splitY
        step_num_X = width // splitX

        total_num = step_num_Y * step_num_X
        finished_num = 1

        # 切割成splitX*splitY小图
        for i in range(step_num_X):
            for j in range(step_num_Y):
                geotrans[0] = leftX + (i * splitX) * geotrans[1]
                geotrans[3] = leftY + (j * splitY) * geotrans[5]
                # print(i, j, geotrans[0], geotrans[3])
                start_X, start_Y = i * splitX, j * splitY
                end_Y = (j + 1) * splitY if (j + 1) * splitY < height else height
                end_X = (i + 1) * splitX if (i + 1) * splitX < width else width

                cur_image = data[:, start_Y:end_Y, start_X:end_X]
                output_filename = '{}_{}.tif'.format(i, j)
                if prefix:
                    output_filename = prefix + '_' + output_filename
                output_filepath = os.path.join(output_folder, output_filename)
                # 写数据
                self.writeImg(output_filepath, proj, geotrans, cur_image)

                print('%d/%d %s %s' % (finished_num, total_num, output_filename, str(cur_image.shape)))
                finished_num += 1

    def tif2other(self, input_fullpath, output_fullpath):
        suffix = output_fullpath.split('.')[-1]
        assert suffix in ['jpg', 'png'], f"{suffix} no in {['jpg', 'png']}"
        # height width channel
        img = cv2.imread(input_fullpath)
        cv2.imwrite(output_fullpath, img)


myRaster = MyRaster()

def test_split_tif():
    # 切割后的长宽
    splitXY = (1000, 1000)
    # 原始的tif文件路径
    tif_fullpath = r''
    # 切割后的tif文件保存路径
    output_folder = r''
    # 文件名前缀
    prefix = ''
    # 要填充的像素
    fill_pixel = 0
    # 切割模式  mode = 1 少的补全 mode = 2 少的不要 mode=3 少的不管
    mode = 1
    myRaster.splitTif(splitXY, tif_fullpath, output_folder, prefix, fill_pixel, mode)


def test_tif2jpg():
    # tif文件夹路径
    tifs_folder = r''
    # 输出的jpg文件夹路径
    jpgs_folder = r''
    # 获取所有tif的路径
    tifs_filename = [filename for filename in os.listdir(tifs_folder) if filename.endswith('.tif')]
    # 遍历所有tif文件
    for tif_filename in tifs_filename:
        # tif文件全路径
        tif_fullpath = os.path.join(tifs_folder, tif_filename)
        # 定义输出文件名
        jpg_filename = tif_filename.split('.')[0] + '.jpg'
        # 输出的jpg的全路径
        jpg_fullpath = os.path.join(jpgs_folder, jpg_filename)
        # tif转jpg
        myRaster.tif2other(tif_fullpath, jpg_fullpath)



def main():
    test_split_tif()

    test_tif2jpg()


if __name__ == '__main__':
    main()
