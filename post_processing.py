
# In this file, we will imlement logics in order manipulate table such that some of the wrong detection is filtered out beforre mapping
# The output csv will be read and diplayed by the mapper.
import pandas as pd

#Loading CSV
complete_csv = pd.read_csv("super_core_data.csv")

#Filtering out in the images without pothole
mask = (complete_csv["class_ids"].str.len() > 2)
pothole_pred_core_df = complete_csv.loc[mask]
print(pothole_pred_core_df)

#Deinging a new column to store metadata next filterign
pothole_pred_core_df["score_filter"] = 0
# This function element all the coordinate which accuracy predition is above given threshold
def score_filter(index, threshold):
  accs = pothole_pred_core_df["scores"][index]
  pothole_pred_core_df["Longitude"][index] = float(pothole_pred_core_df["Longitude"][index])
  pothole_pred_core_df["Latitude"][index] = float(pothole_pred_core_df["Latitude"][index])
  #print(accs, len(accs))
  accs = accs[1:-1]
  accs = accs.split(' ')
  #print(accs)
  for acc in accs:
    #print(acc)
    try:
      if float(acc)>threshold: # this is the filteing value
        pothole_pred_core_df["score_filter"][index] = 1
    except ValueError:
      #print("One", acc)
      a=1
  #print(pothole_pred_core_df["score_filter"][index])

for index, row in pothole_pred_core_df.iterrows():
    threshold = 0.95
    score_filter(index, threshold)

pothole_high_core_df = pothole_pred_core_df[pothole_pred_core_df["score_filter"]== 1]
print(pothole_high_core_df)

#Correcting Latitude with wrong decimal place
pothole_high_core_df["Longitude"] = pothole_high_core_df["Longitude"].apply(lambda x: x/100000000 if x>100 else x )
print(pothole_high_core_df)

#filteing out latitudes/longitude with value of zero. Note for all incorrect optput of coordinate extraction with pytesseract we have replaced it with  a zero
pothole_high_core_df = pothole_high_core_df[pothole_high_core_df["Longitude"]!=0.00]
pothole_high_core_df = pothole_high_core_df[pothole_high_core_df["Latitude"]!=0.00]

# I ahve observed that when a vehicle is slow enough, several photos are taken for same pot hole, which makes
# the red circle very intense, especially at road corners
# avoid this will remove the images which the very similar preceeding image
# reading the file
# We need to sort the dataframe WRT to the image name
df = pothole_high_core_df
df["series"] = 1
df["series"] = df["Image"].apply(lambda x: int(x[17:-4]))
df.sort_values(by=["series"], inplace = True)
df = df.reset_index()

import requests
import os
Root_dir = os.path.abspath(".")
def distance(img1, img2):
  loc1 = os.path.join(Root_dir, r"road_images/" + img1)
  print(loc1)
  loc2 = os.path.join(Root_dir, r"road_images/" + img2)
  print(loc2)
  r= requests.post('https://api.deepai.org/api/image-similarity', files={'image1':open(loc1,'rb'),'image2':open(loc2,'rb'),}, headers={'api-key':'828337ad-e049-4976-8800-53a6a1ec0045'})
  dist_json  = r.json()
  distance = dist_json["output"]["distance"]
  return distance

# adding a similarity value in a  new column agaianst every image
df["similarity"] = 1
df["similarity"][0] = 50
len(df)
for i in range(1,len(df)):
  img1 = df["Image"][i-1]
  #print(img1)
  img2 = df["Image"][i]
  #print(img2)
  a=distance(img1, img2)
  print(i)
  df["similarity"][i] = a


pothole_high_core_df = df
pothole_high_core_df = pothole_high_core_df[pothole_high_core_df["similarity"]> 16] # is the threshold i have set for similarity
pothole_high_core_df.to_csv("final_df.csv")