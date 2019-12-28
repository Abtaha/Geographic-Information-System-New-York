import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
import os


def getMap():
    filesDir = os.getcwd() + "\\new york data\\"

    files = []
    for i in os.listdir(filesDir):
        if i.endswith('.shp'):
            files.append(os.path.join(filesDir, i))

    data = []
    for filename in files:
        file = filename.split("\\")[-1]
        if (file == "gis_osm_roads_free_1.shp" or file == "gis_osm_waterways_free_1.shp" 
            or file == "gis_osm_railways_free_1"):
            data.append(gpd.read_file(filename))

    gdf = pd.concat(data, sort=True)
    return gdf


mapData = getMap()
mapData.plot(linewidth=0.1)

weatherData = []
weatherData.append([pd.read_csv("weather data\\USW00094728.csv"), [40.7789,-73.9692]])
weatherData.append([pd.read_csv("weather data\\USW00014733.csv"), [42.9486,-78.7369]])
weatherData.append([pd.read_csv("weather data\\USW00014735.csv"), [42.7431,-73.8092]])
weatherData.append([pd.read_csv("weather data\\USW00014768.csv"), [43.1167,-77.6767]])

x = [data[1][1] for data in weatherData]
y = [data[1][0] for data in weatherData]


def getTemp(points, weatherData):
    temp = []
    
    for i in range(len(points)):
        df = weatherData[i - 1][0]
        
        tmax_avg = df["tmax"].sum() / len(df["tmax"])
        tmin_avg = df["tmin"].sum() / len(df["tmax"])
        avg = (tmax_avg + tmin_avg) / 2
        # Convert to Celsius
        avg = 5/9 * (avg - 32)
        temp.append(avg)
    
    return np.array(temp)



grid_x, grid_y = np.mgrid[-80:-71:((-71)-(-80))/float(10000), 40:45:(45-40)/float(10000)]

points = np.array([data[1] for data in weatherData])
values = getTemp(points, weatherData)

interpolated = griddata(points, values, (grid_x, grid_y), method='nearest')
plt.imshow(interpolated.T, extent=(-80, -71, 40, 45), origin='lower', cmap="hot")

cmap = plt.cm.hot
custom_lines = [Line2D([0], [0], color=cmap(0.0), lw=4),
                Line2D([0], [0], color=cmap(0.5), lw=4),
                Line2D([0], [0], color=cmap(1.0), lw=4)]
plt.legend(custom_lines, ['Cold', 'Mild', 'Hot'])
plt.show()
