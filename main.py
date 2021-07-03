import sys
import os
import kmeans
import numpy as np
from PIL import Image

def main():
    # reading pixels
    print("Reading pixels...", end="\t")
    if len(sys.argv) != 5:
        print("Arguments are not appropriate")
        return 0
    
    

    imgName = sys.argv[1]
    colornum = int(sys.argv[2])
    max_iter = int(sys.argv[3])
    eps = float(sys.argv[4])
    
    img = Image.open(imgName, 'r')
    width, height = img.size
    h,t = os.path.split(imgName)
    newImageName = t + "_" + sys.argv[2] + "_colors_" + sys.argv[3] + "_epochs_epsilon_" + str(eps) + ".png"
    #print(newImageName)
    get_data_pixels = np.array(list(img.getdata()))
    flatten = [a for sub in get_data_pixels for a in sub]
    p1 = np.array(flatten)
    p2 = [p1]
        
    # read image pixels
    # and have a list in 1 X (width * height) dimensions

    print("DONE")
    
    model = kmeans.KMeans(
        X=np.array(p2),
        n_clusters=colornum,
        max_iterations=max_iter,
        epsilon=eps,
        distance_metric="euclidian"
    )
    print("Fitting...")
    model.fit()    
    print("Fitting... DONE")

    print("Predicting...")
    color1 = (134, 66, 176)
    color2 = (34, 36, 255)
    color3 = (94, 166, 126)
    print(f"Prediction for {color1} is cluster {model.predict(color1)}")
    print(f"Prediction for {color2} is cluster {model.predict(color2)}")
    print(f"Prediction for {color3} is cluster {model.predict(color3)}")

    # replace image pixels with color palette
    willSaved = Image.new(mode="RGB", size=(width,height))
    for i in range(width):
        for j in range(height):
            r,g,b = img.getpixel((i,j))
            index = model.predict((r,g,b))
            newr, newg, newb = model.cluster_centers[index]
            newColor = round(newr), round(newg), round(newb)
            willSaved.putpixel((i,j), newColor)

    # (cluster centers) found in the model
    willSaved.save(newImageName)
    # save the final image


if __name__ == "__main__":
    main()