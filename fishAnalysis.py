import netCDF4
import folium
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import math
from folium import plugins

'''
#fill in blank entries
def fillNull(data):
    mean = np.mean(data)
    for i,val in enumerate(data):
        if val == '--':
            data[i] = mean
    return data

#takes in latitude and longitude coordinates and increments corresponding ICES rectangle by num
def binSample(grid, lat, lon, num=1, latBin=0.5, lonBin=1.0, flipLat=True):
    lonIndex = int((lon + 44)/lonBin)*lonBin
    latIndex = 98 - int((lat - 36)/latBin)*latBin  #flipped so cardinal directions align with grid matrix
    grid[latIndex, lonIndex] = grid[latIndex, lonIndex] + num
'''
    
class Grid:
    #class to store spatial fish and temperature data
    def __init__(self, latBin=0.5, lonBin=1.0, latBounds=(36.0, 85.5), lonBounds=(-44, 70)):
        self.latBin = latBin
        self.lonBin = lonBin
        self.latBounds = latBounds
        self.lonBounds = lonBounds
        self.latRange = latBounds[1] - latBounds[0]
        self.lonRange = lonBounds[1] - lonBounds[0]
        self.shape = (int(self.latRange/latBin), int(self.lonRange/lonBin))
        self.grid = np.empty(self.shape, dtype=object)

        #find center point of each cell and initialize a "Cell" instance, starting from northwest corner
        for i in range(self.shape[0]):
            latCenter = self.latBounds[1] - latBin/2 - i*latBin  #note the latitude is flipped to match cardinal directions with the grid matrix
            for j in range(self.shape[1]):
                lonCenter = self.lonBounds[0] + lonBin/2 + j*lonBin
                self.grid[i,j] = Cell(latCenter, lonCenter, latBin, lonBin)
        

    #method takes dataframe of temp samples and adds the values to tempGrid
    def binTemp(self, tempData):
        numSamples = tempData.shape[0]
        for i in range(numSamples):
            lat, lon, value = np.array(tempData.iloc[i, :])

            #check whether coordinate falls within grid
            if (self.latBounds[0] < lat < self.latBounds[1]) and (self.lonBounds[0] < lon < self.lonBounds[1]):            
                lonIndex = int((lon - self.lonBounds[0]) / self.lonBin)
                latIndex = int((self.latBounds[1] - lat) / self.latBin)
                cell = self.grid[latIndex, lonIndex]
                cell.temps = np.append(cell.temps, value)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                temps = self.grid[i, j].temps
                self.grid[i, j].temp = np.average(temps)

    #method takes dataframe of fish samples and adds their counts to the fishGrid
    def binFish(self, fish):
        numSamples = fish.shape[0]
        for i in range(numSamples):
            lat, lon, num = np.array(fish.iloc[i, :])

            #check whether coordinate falls within grid
            if (self.latBounds[0] < lat < self.latBounds[1]) and (self.lonBounds[0] < lon < self.lonBounds[1]):      
                lonIndex = int((lon - self.lonBounds[0]) / self.lonBin)
                latIndex = int((self.latBounds[1] - lat) / self.latBin)
                self.grid[latIndex, lonIndex].fish += num


    def plotTemp(self, mapObj, color=(0.0, 1.0, 1.0)):
        tempGrid = np.ndarray(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tempGrid[i,j] = self.grid[i,j].temp
        tempGrid = np.nan_to_num(tempGrid) #replace nans with 0                
        tempGrid = tempGrid / np.max(tempGrid)
        colorRGB = colorsys.hsv_to_rgb(color[0], color[1], color[2])
        colorHex = matplotlib.colors.to_hex(colorRGB)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                opacity = tempGrid[i,j]
                folium.Rectangle(
                    bounds=cell.points,
                    fill_color=colorHex,
                    fill=True,
                    fill_opacity=opacity
                    ).add_to(mapObj)
        mapObj.save('tempMap.html')


    def plotFisheries(self, mapObj):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                if cell.fishery:
                    folium.Circle(
                        location=cell.center,
                        radius=1000,
                        color='lightblue',
                        fill=True,
                        fill_opacity=0.7
                        ).add_to(mapObj)

    def plotFish(self, mapObj=None, color=(0.4, 1.0, 1.0)):
        fishGrid = np.ndarray(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                fishGrid[i,j] = self.grid[i,j].fish
        fishGrid = fishGrid / np.max(fishGrid)  #scale between 0 and 1
        if mapObj == None:
            plt.imshow(fishGrid)
            plt.show()
        else:
            threshold = 0.05 #scale values 0 to 0.10 between 0 and 1.0 opacity, set greater values to 1.0
            colorRGB = colorsys.hsv_to_rgb(color[0], color[1], color[2])
            colorHex = matplotlib.colors.to_hex(colorRGB)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    cell = self.grid[i,j]
                    opacity = fishGrid[i,j] / threshold
                    if opacity > 1.0:
                        opacity = 1.0
                    folium.Rectangle(
                        bounds=cell.points,
                        fill_color=colorHex,
                        fill=True,
                        fill_opacity=opacity
                        ).add_to(mapObj)
                    if cell.fishery:
                        folium.Circle(
                            location=cell.center,
                            radius=1000,
                            color='lightblue',
                            fill=True,
                            fill_opacity=0.7
                            ).add_to(mapObj)

    def getTempsAndFish(self):
        xTemps = np.array([])
        yFish = np.array([])
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                if (not math.isnan(cell.temp)) and (cell.fish != 0): #exclude cells without temperature or fish data
                    xTemps = np.append(xTemps, cell.temp)
                    yFish = np.append(yFish, cell.fish)
        return xTemps, yFish

    def getMeanTemp(self):
        xTemps, yFish = self.getTempsAndFish()
        meanTemp = np.dot(xTemps, yFish) / np.sum(yFish)
        return meanTemp
    
    def getStdTemp(self):
        xTemps, yFish = self.getTempsAndFish()
        mean = self.getMeanTemp()
        sqSum = (xTemps - mean)**2
        sqSumMean = np.dot(sqSum, yFish) / np.sum(yFish)
        std = np.sqrt(sqSumMean)
        return std

    #predict fishery locations based on temperature, with optional increase parameter (deg C)
    def guessFisheries(self, change=0):
        meanTemp = self.getMeanTemp()
        stdTemp = self.getStdTemp()
        minTemp = meanTemp - stdTemp
        maxTemp = meanTemp + stdTemp

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                if minTemp < cell.temp + change < maxTemp:
                    cell.fishery = True
                else:
                    cell.fishery = False

    def plotTempVSFish(self):
        xTemps, yFish = self.getTempsAndFish()

        bins = np.ndarray((2, 24)) #rows: (temp,count), columns: 0.5 degree bins
        bins[0,:] = np.arange(4, 16, 0.5) #create bins between 4 and 16 in 0.5 degree intervals

        for i,temp in enumerate(xTemps):
            binIndex = int(2*(temp - 4)) #index where values will go in 'bins' array
            if 0 < binIndex < 24:
                bins[1,binIndex] = bins[1,binIndex] + yFish[i] #increase bin's count by number of fish in sample

        plt.plot(bins[0,:], bins[1,:])
        plt.xlabel('Temperature (deg C)')
        plt.ylabel('Number of fish')
        plt.show()
        
            
                


class Cell:
    def __init__(self, lat, lon, height, width):
        self.lat = lat
        self.lon = lon
        self.center = np.array([lat, lon])
        self.width = width
        self.height = height
        self.fish = 0
        self.temps = np.array([])
        self.temp = 0
        self.fishery = False
        northWest = np.array([self.lat + height/2, self.lon - width/2])
        northEast = np.array([self.lat + height/2, self.lon + width/2])
        southEast = np.array([self.lat - height/2, self.lon + width/2])
        southWest = np.array([self.lat - height/2, self.lon - width/2])
        self.points = np.array([northWest, northEast, southEast, southWest]) #clockwise from northwest corner
        




herringDataFull = pd.read_csv('./herring_2018-19/herring.csv')
herringData = herringDataFull.loc[:, ['Latitude [degrees_north] ', 'Longitude [degrees_east] ', 'NOINP']]

tempDataFull = pd.read_csv('./temp_2012-14.csv')
tempData = tempDataFull.iloc[:, [4,5,8]] #latitude, longitude, temp (deg C)

myGrid = Grid(latBin=0.25, lonBin=0.5, latBounds=(55.0,60.0), lonBounds=(-9.0,2.0))
myGrid.binFish(herringData)
myGrid.binTemp(tempData)

myGrid.plotTempVSFish()

maps = []
for i,change in enumerate([0.0, 0.5, 1.0, 1.5, 2.0]):
    maps.append(folium.Map(location=[56, -3.0], tiles='Stamen Toner', zoom_start=7))
    myGrid.guessFisheries(change)
    myGrid.plotFisheries(maps[i])
    maps[i].save('fisheries_{}deg.html'.format(i))

    

xTemps, yFish = myGrid.getTempsAndFish() #returns order arrays of temperature and fish counts, where both are available

print("Average fish habitat temperature: {} degrees C".format(myGrid.getMeanTemp()))
print("Fish habitat temperature standard deviation: {} degrees C".format(myGrid.getStdTemp()))


'''
plt.scatter(xTemps, yFish)
plt.xlabel('Temperature (deg C)')
plt.ylabel('Number of Herring')
plt.title('Abundance of Herring as Function of Temperature')
plt.show()
'''     









