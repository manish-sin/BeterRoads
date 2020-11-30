#This file will generate an html file with pothole coodinates


#Here will define the templet of stsarting portion of the HTML file
head = """
<html>
<head>
<meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>Google Maps - gmplot</title>
<script type="text/javascript" src="https://maps.googleapis.com/maps/api/js?libraries=visualization"></script>
<script type="text/javascript">
    function initialize() {
        var map = new google.maps.Map(document.getElementById("map_canvas"), {
            zoom: 12,
            center: new google.maps.LatLng(28.393737899999994, 77.03196649)
        });"""

#This sectins templet for puting a mark at specific location, this setion of code is placed in HTML for each location
def markers(marks, lat, long, linker):
    var = marks
    marker1= "const %s = new google.maps.Circle({strokeColor: '#FF0000',strokeOpacity: 1.0, strokeWeight: 1,fillColor: '#FF0000', fillOpacity: 0.3, map: map, center:" %(var)
    marker2 = "new google.maps.LatLng(%f,"%(lat)
    marker3= "%f), radius: 0.5});"%(long)
    link = "%s.addListener"%(var)
    link1 ='("click", () => {window.open("pot_holes_detected/%s");});'%(linker)
    return marker1, marker2, marker3, link, link1

#This section is templet of last potion of HTML file , it also contains layout of the webpage
tail = """    
    }
</script>
</head>
<h1> This Map hilights pot holes on your Roads</h1>
<h5> Click on the red dots to see the pot hole </h5>    

<h5> Source code:https://github.com/manish-sin/pothole_heatmap</h5>
<body style="margin:0px; padding:0px;" onload="initialize()">
    <div id="map_canvas" style="width: 100%; height: 84.5%;" />
</body>
</html>"""
import pandas as pd
#loading the final_df.csv for ploting
core_df = pd.read_csv("final_df.csv")
#sttting the image names as index for simplifided iteration
core_df=core_df.set_index("Image")
#this variable is used to give each point a unique value
i=0
#creating the HTML file
with open('pothole_map.html', 'w') as f:
    f.write(head)
    space = """
    """

    for img in core_df.index:
        row = core_df.loc[img]
        print(row)
        marker = "marker%i"%(i)
        marker1, marker2, marker3, link, link1 =markers(marker, row[3], row[4], img)
        i=i+1

        f.write(space)
        f.write(marker1)


        f.write(marker2)

        f.write(marker3)
        f.write(space)

        f.write(link)
        print(link1)
        f.write(link1)
        f.write(space)
    f.write(tail)