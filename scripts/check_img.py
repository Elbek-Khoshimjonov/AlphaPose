import os
import cv2
import argparse

'''
깨진 이미지가 있는지 없는지 확인하는 코드
'''

def main():
    parser = parser = argparse.ArgumentParser(description='check image')
    parser.add_argument('--img_path', help="image path")
    args = parser.parse_args()

    img_path = os.path.abspath(args.img_path)
    image_list = os.listdir(img_path)

    error_img = []
    total_img = len(image_list)
    for num, i in enumerate(image_list):
        image_path = img_path + "/" + i
        img = cv2.imread(image_path)
        print(f'{num}/{total_img}')
        try:
            img_size = img.shape
        except:
            error_img.append(i)
    
    print(error_img)
    
if __name__ == "__main__":
    main()