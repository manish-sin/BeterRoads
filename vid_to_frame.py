#Step 1.
#Take a video frame placed in \road_survey_vid\upload_folder. it strips the video into frames
import cv2
import os
Root_dir = os.path.abspath(".")

#defing path of folder from which videos are/can be stored
upload_folder_dir = os.path.join(Root_dir, r"road_survey_vid\upload_folder" )
vid_list= os.listdir(upload_folder_dir)
print(vid_list)

#selecting the first video in that folder
upload_vid_loc = os.path.join(upload_folder_dir, vid_list[0])
print(upload_vid_loc)

#creating instance of CV2 video
vidcap = cv2.VideoCapture(upload_vid_loc)

#Checking if video was read properly or not. sucess variable returns boolean value
success,image = vidcap.read()

#Defining a varible just keep count of frame no. This numbers will be added to each frame to give it a unique name
count = 0

#location of folder in which the frames of video will be stored
path = os.path.join(Root_dir, r"frames")

#Reading the FPS of video. This value is not used anywhere, it is just
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
print(fps)

# code below will extract all the frames in the video
while success:
  cv2.imwrite(os.path.join(path, "{}_frame{}.jpg".format(vid_list[0],count)), image)     # save frame as JPEG file
  print("{}_frame{}.jpg".format(vid_list[0],count))
  success,image = vidcap.read()
  count += 1 # this stes the frequency of capturing frames
  print('Read a new frame at ', count, " :", success)

