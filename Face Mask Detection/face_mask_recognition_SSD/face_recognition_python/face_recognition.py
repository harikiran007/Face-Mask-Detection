from PIL import Image
import face_recognition

picPosition='/Users/lskmac/computer/face_recognition/mingren4.jpeg'
peopleImage = face_recognition.load_image_file(picPosition)
realImage = Image.open(picPosition, 'r')


face_marks = face_recognition.face_landmarks(peopleImage,  model="large")

img = Image.open('/Users/lskmac/computer/face_recognition/mask/mask3.png', 'r')

bg_w, bg_h = img.size

# height=face_marks[0]['chin'][10][1]-face_marks[0]['nose_bridge'][0][1]
bottom=face_marks[0]['chin'][8][1]
top=face_marks[0]['nose_bridge'][1][1]
left=face_marks[0]['left_eyebrow'][0][0]
right=face_marks[0]['right_eyebrow'][4][0]
print(face_marks)
width=right-left
left=int(left-width*0.2)
right=int(right+width*0.2)



img = img.resize((right-left,int(bottom-top)), Image.ANTIALIAS)

realImage.paste(img, (left,top,right,bottom),img.convert('RGBA'))
realImage.show()

face_image = peopleImage[top:bottom, left:right]
pil_image = Image.fromarray(face_image)
pil_image.show()

