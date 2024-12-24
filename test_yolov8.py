from ultralytics import YOLO

import cv2
import numpy as np
import math


def SortPoint(four_points):


    p1 = np.array([four_points[0][0], four_points[0][1]])
    p2 = np.array([four_points[1][0], four_points[1][1]])
    p3 = np.array([four_points[2][0], four_points[2][1]])
    p4 = np.array([four_points[3][0], four_points[3][1]])

    #### find postion ####

    Cl = (p1 + p2 + p3 + p4) / 4

    dv1 = p1 - Cl
    dv2 = p2 - Cl
    dv3 = p3 - Cl
    dv4 = p4 - Cl


    a1 = math.atan2(dv1[1], dv1[0]) + math.pi
    a2 = math.atan2(dv2[1], dv2[0]) + math.pi
    a3 = math.atan2(dv3[1], dv3[0]) + math.pi
    a4 = math.atan2(dv4[1], dv4[0]) + math.pi

    a = [int(a1), int(a2), int(a3), int(a4)]

    sc = sorted(range(len(a)), key=lambda k: a[k])



    rect = np.zeros((4, 2), dtype=np.float32)

    rect[0][0]=four_points[sc[0]][0]
    rect[0][1] = four_points[sc[0]][1]

    rect[1][0] = four_points[sc[1]][0]
    rect[1][1] = four_points[sc[1]][1]

    rect[2][0] = four_points[sc[2]][0]
    rect[2][1] = four_points[sc[2]][1]

    rect[3][0] = four_points[sc[3]][0]
    rect[3][1] = four_points[sc[3]][1]



    return (rect)

def FindContour(img, mask):


    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 3)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))

    mask = cv2.dilate(mask, k1)
    mask = cv2.dilate(mask, k1)


    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    max_area = 0
    max_contour = None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)  # Caculate Contour's area
        if area > max_area:
            max_area = area
            max_contour = cnt



    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    cv2.polylines(img, [approx], thickness=6, color=(255, 0, 0), isClosed=True)


    points = np.zeros((4, 2), dtype=np.float32)

    if (len(approx) == 4):

        for i in range(len(approx)):
            x = approx[i][0][0]
            y = approx[i][0][1]
            points[i][0] = x
            points[i][1] = y

    else:

        for i in range(0, 3):
            points[i][0] = 0
            points[i][1] = 0

        print('Cant find 4 points')

    return (points)


def main():
    model_path = 'weights/train1.pt'

    basepath = "test_image/"
    image_path = basepath + '/3.jpg'
    save_final = basepath + 'result.png'
    save_cont = basepath + 'cont.png'

    img = cv2.imread(image_path)
    H, W, _ = img.shape

    width = 300
    height = 400
    result_img = img.copy()
    crop = img.copy()
    out_img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 8)
    model = YOLO(model_path)

    mask_count = 0

    # Load model
    results = model(img)

    for result in results:
        for j, mask in enumerate(result.masks.data):

            if mask is None:

                print("no images detection")

            else:
                mask_count = 1
                mask = mask.cpu().numpy() * 255
                mask_img = cv2.resize(mask, (W, H))

    if (mask_count == 0):

        print("no images detection")

    else:

        mask_img = np.expand_dims(mask_img, axis=-1)

        kernel2 = np.ones((5, 5), np.uint8)
        mask_img = cv2.dilate(mask_img, kernel2, iterations=3)

        cv2.namedWindow('contours', flags=cv2.WINDOW_NORMAL)
        cv2.imshow('contours', mask_img)

        final_corner = FindContour(result_img, mask_img)

        if (final_corner[0][0] == 0):

            print("Cant find 4 points")
        else:
            final_corner = np.float32(final_corner)
            final_corner = SortPoint(final_corner)
            print("Cotour's Four points:", final_corner)

            cv2.circle(result_img, (int(final_corner[0][0]), int(final_corner[0][1])), 8, (0, 0, 255), -1)
            cv2.circle(result_img, (int(final_corner[1][0]), int(final_corner[1][1])), 8, (0, 0, 255), -1)
            cv2.circle(result_img, (int(final_corner[2][0]), int(final_corner[2][1])), 8, (0, 0, 255), -1)
            cv2.circle(result_img, (int(final_corner[3][0]), int(final_corner[3][1])), 8, (0, 0, 255), -1)

            cv2.namedWindow('result_img', flags=cv2.WINDOW_NORMAL)
            cv2.imshow('result_img', result_img)
            cv2.imwrite(save_cont, result_img)

            # 목적지 좌표 설정
            pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

            # 코너를 np.float32로 변환
            final_corner = np.array([[int(final_corner[i][0]), int(final_corner[i][1])] for i in range(4)],
                                    dtype=np.float32)

            h, status = cv2.findHomography(final_corner, pts_dst)

            warped_img = cv2.warpPerspective(out_img, h, (width, height), cv2.INTER_AREA)
            print('warped_im:', warped_img.shape)

            cv2.namedWindow('warped_img', flags=cv2.WINDOW_NORMAL)
            cv2.imshow('warped_img', warped_img)
            cv2.imwrite(save_final, warped_img)

    cv2.waitKey(0)


if __name__ == "__main__":
  main()