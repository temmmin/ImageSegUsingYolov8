from xml.dom import minidom
import os
import glob
import xml.etree.ElementTree as ET


for fname in glob.glob("xml/*.xml"):
        xmldoc = minidom.parse(fname)
        fname_out = (fname[:-4]+'.txt')

        with open(fname_out, "w") as f:

                itemlist = xmldoc.getElementsByTagName('object')

                size = xmldoc.getElementsByTagName('imagesize')[0]

                width = int((size.getElementsByTagName('ncols')[0]).firstChild.data)
                height = int((size.getElementsByTagName('nrows')[0]).firstChild.data)

                for item in itemlist:


                        xmin = (
                        (item.getElementsByTagName('pt')[0]).getElementsByTagName('x')[0]).firstChild.data

                        #print(xmin)


def convert_polygon_to_yolov8(polygon, image_width, image_height):
    yolov8_format = []
    pp=[]


    for point in polygon:
        x, y = point

        pp.append(x / image_width)
        pp.append(y / image_height)

    yolov8_format.append(pp)

    return (yolov8_format,pp)

# 함수: Pascal VOC 형식의 XML 파일을 YOLOv8 형식의 텍스트 파일로 변환
def convert_xml_to_yolov8(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find("./imagesize/ncols").text)
    image_height = int(root.find("./imagesize/nrows").text)

    objects = root.findall("./object")
    for obj in objects:
        polygon_elements = obj.findall("./polygon/pt")
        polygon = [(float(pt.find("x").text), float(pt.find("y").text)) for pt in polygon_elements]

        yolov8_format,pp= convert_polygon_to_yolov8(polygon, image_width, image_height)

        #print('i:',yolov8_format)

        output_file_path = os.path.join(output_dir, "{}.txt".format(os.path.splitext(root.find("filename").text)[0]))

        with open(output_file_path, 'w') as f:

            for pp in yolov8_format:

                for p_, p in enumerate(pp):

                    print('i:', p)
                    # print('i:', p_, ",", "coord:", p)

                    if p_ == len(pp) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))


        # with open(output_file_path, 'w') as f:
        #
        #     for pp in yolov8_format:
        #
        #         for p_, p in enumerate(polygon):
        #
        #             print('i:', p)
        #             # print('i:', p_, ",", "coord:", p)
        #
        #             if p_ == len(polygon) - 1:
        #                 f.write('{}\n'.format(p))
        #             elif p_ == 0:
        #                 f.write('0 {} '.format(p))
        #             else:
        #                 f.write('{} '.format(p))

        # with open(output_file_path, 'w') as f:
        #
        #     for i, coord in enumerate(yolov8_format):
        #         # if i % 2 == 0:
        #         #     f.write("{} ".format(coord))
        #         # else:
        #         #     f.write("{}\n".format(coord))
        #
        #         print('i:',i, ",","coord:",coord)



            # for poly in polygon:
            #     for p_, p in enumerate(polygon):
            #         if p_ == len(poly) - 1:
            #             f.write('{}\n'.format(p))
            #         elif p_ == 0:
            #             f.write('0 {} '.format(p))
            #         else:
            #             f.write('{} '.format(p))


# change to yuor dataset xml & txt path

xml_folder = "D:/dataset/segmentation/xml/"
output_folder ="D:/dataset/segmentation/txt/"

for xml_file_name in os.listdir(xml_folder):
    if xml_file_name.endswith(".xml"):
        xml_file_path = os.path.join(xml_folder, xml_file_name)
        convert_xml_to_yolov8(xml_file_path, output_folder)
