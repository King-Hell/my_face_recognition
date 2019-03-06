import face_recognition
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np

'''从摄像头中识别人脸，基于face_recognition示例修改'''

def change_cv2_draw(image,strs,local,sizes):
    '''在图片中显示中文'''
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("msyh.ttc",sizes, encoding="utf-8")#微软雅黑
    colour=(255,255,255)
    draw.text(local, strs, colour, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image

#获取摄像头数据
video_capture = cv2.VideoCapture(0)#0为默认摄像头

#加载人脸图片
test_image = face_recognition.load_image_file("奥巴马.jpg")
test_face_encoding = face_recognition.face_encodings(test_image)[0]

#生成人脸编码数组
known_face_encodings = [
    test_face_encoding
]
known_face_names = [
    "奥巴马"
]

#初始化
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #抓取一帧图片
    ret, frame = video_capture.read()
    #缩小图片
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #BGR转RGB
    rgb_small_frame = small_frame[:, :, ::-1]
    #处理视频帧
    if process_this_frame:
        #确定人脸位置并编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            #与已有人脸编码比对
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            #matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.39)
            name = "未知"
            #如果有多个匹配采用第一个
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame

    #结果显示
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        #重新放大图片
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #绘制矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #绘制人名
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        frame=change_cv2_draw(frame,name,(left + 10, bottom - 40),30
    #显示结果图片
    cv2.imshow('Video', frame)
    #ESC键退出
    if cv2.waitKey(1) & 0xFF == 27:#27为ESC键
        break
#释放
video_capture.release()
cv2.destroyAllWindows()