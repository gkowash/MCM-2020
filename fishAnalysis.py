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
            dateStr, lat, lon, value = np.array(tempData.iloc[i, :])
            monthIndex = int(dateStr[5:7]) - 1 #Jan:0 , ..., Dec: 11

            #check whether coordinate falls within grid
            if (self.latBounds[0] < lat < self.latBounds[1]) and (self.lonBounds[0] < lon < self.lonBounds[1]):            
                lonIndex = int((lon - self.lonBounds[0]) / self.lonBin)
                latIndex = int((self.latBounds[1] - lat) / self.latBin)
                cell = self.grid[latIndex, lonIndex]
                cell.temps[monthIndex].append(value)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                temps = cell.temps
                for k in range(0,12):
                    cell.temp[k] = np.mean(np.array(temps[k])) #this is a syntactical trainwreck 

    #method takes dataframe of fish samples and adds their counts to the fishGrid
    def binFish(self, fish, species):
        numSamples = fish.shape[0]
        for i in range(numSamples):
            dateStr, lat, lon, num = np.array(fish.iloc[i, :])
            monthIndex = int(dateStr[3:5]) - 1 #Jan: 0, ..., Dec: 11

            #check whether coordinate falls within grid
            if (self.latBounds[0] < lat < self.latBounds[1]) and (self.lonBounds[0] < lon < self.lonBounds[1]):      
                lonIndex = int((lon - self.lonBounds[0]) / self.lonBin)
                latIndex = int((self.latBounds[1] - lat) / self.latBin)
                cell = self.grid[latIndex, lonIndex]
                if species == 'herring':
                    cell.herring[monthIndex] = cell.herring[monthIndex] + num
                elif species == 'mackerel':
                    cell.mackerel[monthIndex] = cell.mackerel[monthIndex] + num


    #not working, needs to be redesigned
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


    def plotFisheries(self, herrMap, mackMap, seasonIndex, radius=5000):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                if cell.herrFishery[seasonIndex]:
                    folium.Circle(
                        location=cell.center,
                        radius=radius,
                        color='lightblue',
                        fill=True,
                        fill_opacity=0.7
                        ).add_to(herrMap)
                if cell.mackFishery[seasonIndex]:
                    folium.Circle(
                        location=cell.center,
                        radius=radius,
                        color='red',
                        fill=True,
                        fill_opacity=0.7
                        ).add_to(mackMap)
                
    #also maybe does not work?
    def plotFish(self, species, mapObj=None, color=(0.4, 1.0, 1.0)):
        fishGrid = np.ndarray(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if species.lower() == 'herring':
                    fishGrid[i,j] = self.grid[i,j].herring
                elif species.lower() == 'mackerel':
                    fishGrid[i,j] = self.grid[i,j].mackerel
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
    @staticmethod
    def plotPorts(mapObj, radius):
        ports = ((60.15, -1.15), (57.51, -1.78), (57.90, -5.16))
        for coords in ports:
            folium.Circle(
                location=coords,
                radius=radius,
                color='green',
                fill=True,
                fill_opacity=0.2,
                ).add_to(mapObj)

    #returns ordered arrays of temperatures, herring counts, and mackerel counts, excluding missing data
    def getTempsAndFish(self):
        herrTemps = np.array([])
        mackTemps = np.array([])
        herring = np.array([])
        mackerel = np.array([])
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                for k in range(12):
                    if (not math.isnan(cell.temp[k])):
                        if (cell.herring[k] != 0): #exclude cells without temperature or fish data
                            herrTemps = np.append(herrTemps, cell.temp[k])
                            herring = np.append(herring, cell.herring[k])
                        if (cell.mackerel[k] != 0):
                            mackTemps = np.append(mackTemps, cell.temp[k])
                            mackerel = np.append(mackerel, cell.mackerel[k])
        return herrTemps, mackTemps, herring, mackerel

    #find average temperature where each species is caught (weighted by number caught)
    def getMeanTemps(self):
        herrTemps, mackTemps, herring, mackerel = self.getTempsAndFish()
        herrMeanTemp = np.dot(herrTemps, herring) / np.sum(herring)
        mackMeanTemp = np.dot(mackTemps, mackerel) / np.sum(mackerel)
        return herrMeanTemp, mackMeanTemp

    #find standard deviations of temperature distributions (weighted by number caught)
    def getStdTemps(self):
        herrTemps, mackTemps, herring, mackerel = self.getTempsAndFish()
        herrMeanTemp, mackMeanTemp = self.getMeanTemps()
        herrSq = (herrTemps - herrMeanTemp)**2
        mackSq = (mackTemps - mackMeanTemp)**2
        herrSqMean = np.dot(herrSq, herring) / np.sum(herring)
        mackSqMean = np.dot(mackSq, mackerel) / np.sum(mackerel)
        herrStdTemp = np.sqrt(herrSqMean)
        mackStdTemp = np.sqrt(mackSqMean)
        return herrStdTemp, mackStdTemp

    #predict fishery locations based on temperature, with optional increase parameter (deg C)
    def guessFisheries(self, change=0, intervalSize=1):
        herrMeanTemp, mackMeanTemp = self.getMeanTemps()
        herrStdTemp, mackStdTemp = self.getStdTemps()
        herrMinTemp = herrMeanTemp - herrStdTemp*intervalSize #intervalSize: 1 for ~70%, 2 for ~95% of landings contained in interval
        herrMaxTemp = herrMeanTemp + herrStdTemp*intervalSize
        mackMinTemp = mackMeanTemp - mackStdTemp*intervalSize
        mackMaxTemp = mackMeanTemp + mackStdTemp*intervalSize

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = self.grid[i,j]
                for seasonIndex in range(4):
                    if herrMinTemp < np.mean(cell.temp[3*seasonIndex: 3*(seasonIndex+1)]) + change < herrMaxTemp:
                        cell.herrFishery[seasonIndex] = True
                    else:
                        cell.herrFishery[seasonIndex] = False
                    if mackMinTemp < np.mean(cell.temp[3*seasonIndex: 3*(seasonIndex+1)]) + change < mackMaxTemp:
                        cell.mackFishery[seasonIndex] = True
                    else:
                        cell.mackFishery[seasonIndex] = False

    def plotTempVSFish(self):
        herrTemps, mackTemps, herring, mackerel = self.getTempsAndFish()

        herrBins = np.ndarray((2, 24)) #rows: (temp,count), columns: 0.5 degree bins
        mackBins = np.ndarray((2, 24))
        herrBins[0,:] = np.arange(4, 16, 0.5) #create bins between 4 and 16 in 0.5 degree intervals
        mackBins[0,:] = np.arange(4, 16, 0.5)

        for i,temp in enumerate(herrTemps):
            herrBinIndex = int(2*(temp - 4)) #index where particular value will go in 'bins' array
            if 0 < herrBinIndex < 24:
                herrBins[1,herrBinIndex] = herrBins[1,herrBinIndex] + herring[i] #increase bin's count by number of fish in sample
        for i,temp in enumerate(mackTemps):
            mackBinIndex = int(2*(temp -4))
            if 0 < mackBinIndex < 24:
                mackBins[1,mackBinIndex] = mackBins[1,mackBinIndex] + mackerel[i]

        plt.plot(herrBins[0,:], herrBins[1,:])
        plt.xlabel('Temperature (deg C)')
        plt.ylabel('Number of herring')
        plt.title('Herring landing as function of water temperature')
        plt.show()

        plt.plot(mackBins[0,:], mackBins[1,:])
        plt.xlabel('Temperature (deg C)')
        plt.ylabel('Number of mackerel')
        plt.title('Mackerel landing as function of water temperature')
        plt.show()
        
            
                


class Cell:
    def __init__(self, lat, lon, height, width):
        self.lat = lat
        self.lon = lon
        self.center = np.array([lat, lon])
        self.width = width
        self.height = height
        self.herring = np.zeros(12)
        self.mackerel = np.zeros(12)
        self.temps = [[],[],[],[],[],[],[],[],[],[],[],[]] #is there an even less elegant way to handle this?
        self.temp = np.zeros(12)
        self.herrFishery = [False, False, False, False]
        self.mackFishery = [False, False, False, False]
        northWest = np.array([self.lat + height/2, self.lon - width/2])
        northEast = np.array([self.lat + height/2, self.lon + width/2])
        southEast = np.array([self.lat - height/2, self.lon + width/2])
        southWest = np.array([self.lat - height/2, self.lon - width/2])
        self.points = np.array([northWest, northEast, southEast, southWest]) #clockwise from northwest corner
        



#herringDataFull = pd.read_csv('./data/herring_2015-19.csv')
herringDataFull = pd.read_csv('./herring_2018-19/herring.csv')
herringData = herringDataFull.loc[:, ['Datetime', 'Latitude [degrees_north] ', 'Longitude [degrees_east] ', 'NOINP']]

#mackerelDataFull = pd.read_csv('./data/mackerel_2015-19.csv')
mackerelDataFull = pd.read_csv('./mackerel_2012-14/mackerel.csv')
mackerelData = mackerelDataFull.loc[:, ['Datetime', 'Latitude [degrees_north] ', 'Longitude [degrees_east] ', 'NOINP']]

#tempDataFull = pd.read_csv('./data/temp_2015-19.csv')
tempDataFull = pd.read_csv('./temp_2012-14.csv')
tempData = tempDataFull.iloc[:, [3,4,5,8]] #date, latitude, longitude, temp (deg C)

myGrid = Grid(latBin=0.20, lonBin=0.25, latBounds=(55.0,65.0), lonBounds=(-9.0,5.0))
myGrid.binFish(herringData, species='herring')
myGrid.binFish(mackerelData, species='mackerel')
myGrid.binTemp(tempData)

myGrid.plotTempVSFish()

herrMaps = [[],[],[],[]] #one set of maps for each season
mackMaps = [[],[],[],[]]
for seasonIndex in range(4):
    for i,change in enumerate([0.0, 1.0, 1.5, 2.0]):
        herrMaps[seasonIndex].append(folium.Map(location=[58.5, -3.0], tiles='Stamen Toner', zoom_start=7))
        mackMaps[seasonIndex].append(folium.Map(location=[58.5, -3.0], tiles='Stamen Toner', zoom_start=7))
        myGrid.guessFisheries(change, intervalSize=2.3)
        myGrid.plotFisheries(herrMaps[seasonIndex][i], mackMaps[seasonIndex][i], seasonIndex)
        Grid.plotPorts(herrMaps[seasonIndex][i], radius=120000)
        Grid.plotPorts(mackMaps[seasonIndex][i], radius=120000)
    
        herrMaps[seasonIndex][i].save('./seasonal maps/herring_season{}_change{}.html'.format(seasonIndex, i))
        mackMaps[seasonIndex][i].save('./seasonal maps/mackerel_season{}_change{}.html'.format(seasonIndex, i))

herrMeanTemp, mackMeanTemp = myGrid.getMeanTemps()
herrStdTemp, mackStdTemp = myGrid.getStdTemps()



print("Average herring, mackerel habitat temperatures: {}, {} degrees C".format(herrMeanTemp, mackMeanTemp))
print("Fish habitat temperature standard deviation: {}, {} degrees C".format(herrStdTemp, mackStdTemp))


'''
plt.scatter(xTemps, yFish)
plt.xlabel('Temperature (deg C)')
plt.ylabel('Number of Herring')
plt.title('Abundance of Herring as Function of Temperature')
plt.show()
'''     









