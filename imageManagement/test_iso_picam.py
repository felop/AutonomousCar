import io, picamera, time, cv2

camera = picamera.PiCamera()
u = int(input("u = "))
camera.resolution = (80,64)
camera.iso = 100
camera.framerate = 80
camera.start_preview()
isos = []
for i in range(10):
    iso = 100*(i+1)
    time.sleep(0.1)
    print("pic nb "+str(u+i)+" , iso : "+str(iso))
    camera.iso = iso
    title = "iso/"+str(iso)+"_"+str(u)+".png"
    camera.capture(title , use_video_port=True)
    img = cv2.imread(title , cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(1,1))
    isos.append(img[0][0])
print(iso)
