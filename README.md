# FaceRecognition
Simple Personal Face Recognition Project utilizing a Siamese Network enabled Face Recognition Library.


Upon running this on CPU device, the frame processing was very slow;Therefore, multi-threading of captured frames were implemented on top of cv2.videocapture.

Recognized face was recorded on .csv file - Upon detecting a face on a frame, the name of that person is recorded (just once)
