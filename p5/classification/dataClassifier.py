# dataClassifier.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
import search
from pacman import GameState

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features


"https://github.com/anthony-niklas/cs188/blob/master/p5/dataClassifier.py"
"https://github.com/naderm/cs188/blob/master/p5/classification/dataClassifier.py"

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
      aspect ration: 1 if > 2, 0 otherwise
      topHeavy, bottomHeavy: true if > 60 percent of black in that region
      #regions:  1, 2, or more than three
    ##
    """

    #Get basic features
    features = basicFeatureExtractorDigit(datum)

    #use floodfill to count all the whitespace surrounding the character
    #if the flood fill whitespace is less than 98% of the total whitespace (allows for noise),
    #then there are loops in the figure   


    bounding_box = getBoundingBox(datum)
    blackcount = basicFeatureExtractorDigit(datum).totalCount()
    black_in_top = blackInTop(datum, bounding_box, blackcount)
    black_in_left = blackInLeft(datum, bounding_box, blackcount)
    leftprobe, rightprobe = getHorizontalProbes(datum)
    bottomprobe, topprobe = getVerticalProbes(datum)
    regions = getContiguousRegions(datum)
    has_loops = hasLoops(datum, blackcount)
    aspectratio = getAspectRatio(bounding_box)
    sidesmoothness = getSideSmoothness(leftprobe, rightprobe, bottomprobe, topprobe) #this has potential but it needs work...


    #features["hasLoops"] = has_loops
    features["aspectratio_gt_2"] = 1 if aspectratio > 2 else 0
    features["isTopHeavy"] = 1 if black_in_top > 0.6 else 0
    features["isLeftHeavy"] = 1 if black_in_left > 0.6 else 0
    features["regions_1"] = 1 if regions == 1 else 0
    features["regions_2"] = 1 if regions == 2 else 0
    features["regions_3+"] = 1 if regions >= 3 else 0

    
    #print datum
    #print sidesmoothness
    """
    print bounding_box
    print black_in_top
    print black_in_left
    print regions
    print has_loops
    print aspectratio
    """

    return features


def getSideSmoothness(leftprobe, rightprobe, bottomprobe, topprobe):

  leftdrops = 0
  rightdrops = 0
  topdrops = 0
  bottomdrops = 0

  for y in range(DIGIT_DATUM_HEIGHT - 1):
    leftdrops += 1 if abs(leftprobe[y] - leftprobe[y+1]) > 2 else 0
    rightdrops += 1 if abs(rightprobe[y] - rightprobe[y+1]) > 2 else 0

  for x in range(DIGIT_DATUM_HEIGHT - 1):
    topdrops += 1 if abs(topprobe[x] - topprobe[x+1]) > 2 else 0
    bottomdrops += 1 if abs(bottomprobe[x] - bottomprobe[x+1]) > 2 else 0

  return {"left": leftdrops, "right": rightdrops, "bottom": bottomdrops, "top": topdrops}

def floodfillCoords(x, y, datum, region): #region is a set
    if (x,y) in region:
      return
    if datum.getPixel(x,y) > 1:
      return
    region.add((x,y))
    
    if x > 0:
      floodfillCoords(x - 1, y, datum, region)
    if x < DIGIT_DATUM_WIDTH - 1:
      floodfillCoords(x + 1, y, datum, region)
    if y > 0:
      floodfillCoords(x, y - 1, datum, region)    
    if y < DIGIT_DATUM_HEIGHT - 1:
      floodfillCoords(x, y + 1, datum, region)

def getContiguousRegions(datum):
    regions = []
    regions.append(set())
    floodfillCoords(0,0,datum,regions[0])

    for x in range(DIGIT_DATUM_WIDTH):
      for y in range(DIGIT_DATUM_HEIGHT):
        if datum.getPixel(x,y) <= 1 and notInAnySet((x,y), regions):
          newRegion = set()
          floodfillCoords(x,y,datum,newRegion)
          regions.append(newRegion)

    return len([r for r in regions if len(r) > 1])

def notInAnySet(value, setlist):
    for _set in setlist:
      if value in _set:
        return False
    return True

def floodfill(x, y, datum, tracker):
    if tracker[(x,y)] == 1:
      return
    if datum.getPixel(x,y) > 0:
      return
    tracker[(x,y)] = 1
    
    if x > 0:
      floodfill(x - 1, y, datum, tracker)
    if x < DIGIT_DATUM_WIDTH - 1:
      floodfill(x + 1, y, datum, tracker)
    if y > 0:
      floodfill(x, y - 1, datum, tracker)    
    if y < DIGIT_DATUM_HEIGHT - 1:
      floodfill(x, y + 1, datum, tracker)

def getHorizontalProbes(datum):
    leftprobe = [0 for _ in range(DIGIT_DATUM_HEIGHT)]
    rightprobe = [0 for _ in range(DIGIT_DATUM_HEIGHT)]
    for y in range(DIGIT_DATUM_HEIGHT):
      firstblack = None
      lastblack = None
      for x in range(DIGIT_DATUM_WIDTH):
          if datum.getPixel(x, y) > 0:
              if firstblack == None:
                firstblack = x
              lastblack = x  
      leftprobe[y] = DIGIT_DATUM_WIDTH if firstblack == None else firstblack
      rightprobe[y] = 0 if firstblack == None else lastblack

    return [leftprobe, rightprobe]


def getVerticalProbes(datum):
    bottomprobe = [0 for _ in range(DIGIT_DATUM_HEIGHT)]
    topprobe = [0 for _ in range(DIGIT_DATUM_HEIGHT)]
    for x in range(DIGIT_DATUM_WIDTH):
      firstblack = None
      lastblack = None
      for y in range(DIGIT_DATUM_HEIGHT):
          if datum.getPixel(x, y) > 0:
              if firstblack == None:
                firstblack = y
              lastblack = y  
      bottomprobe[y] = DIGIT_DATUM_WIDTH if firstblack == None else firstblack
      topprobe[y] = 0 if firstblack == None else lastblack
      
    return [bottomprobe, topprobe]


def hasLoops(datum, blackcount):
    whitespace = util.Counter()
    floodfill(0,0,datum,whitespace)
    totalwhite = DIGIT_DATUM_HEIGHT * DIGIT_DATUM_WIDTH - blackcount 
    hasLoops = 1 if 1.0 * whitespace.totalCount()/totalwhite < 0.98 else 0 
    return hasLoops


def getAspectRatio(bb):
    #get aspect ratio and bounding box
    minx = bb["minx"]
    miny = bb["miny"]
    maxx = bb["maxx"]
    maxy = bb["maxy"]
    totalwidth = max(maxx - minx,1)
    totalheight = max(maxy - miny,1)
    return (1.0 * totalheight) / totalwidth


def getBoundingBox(datum):
  bb = util.Counter()
  minx = float("inf")
  miny = float("inf")
  maxx = float("-inf")
  maxy = float("-inf")
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x,y) >= 1:
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)
  bb["minx"] = minx
  bb["miny"] = miny
  bb["maxx"] = maxx
  bb["maxy"] = maxy
  return bb


def blackInTop(datum, bb, totalblack):
    minx = bb["minx"]
    miny = bb["miny"]
    maxx = bb["maxx"]
    maxy = bb["maxy"]
    top = 0.0
    for x in range(minx, maxx + 1):
      for y in range(miny, (miny + maxy)/2):
        if datum.getPixel(x,y) > 0:
          top += 1 
    return top / totalblack

def blackInLeft(datum, bb, totalblack):     
    minx = bb["minx"]
    miny = bb["miny"]
    maxx = bb["maxx"]
    maxy = bb["maxy"]  
    left = 0.0
    for x in range(minx, (minx + maxx)/2):
      for y in range(miny, maxy + 1):
        if datum.getPixel(x,y) > 0:
          left += 1 
    return left / totalblack


def enhancedFeatureExtractorDigitOld(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    #a = datum.getPixels()
    

    whitespace = util.Counter()
    floodfill(0,0,datum,whitespace)

    averagex = 0.0
    averagey = 0.0

    features = util.Counter()
    totalblack = 0
    
    minthickness = float("inf")
    maxthickness = float("-inf")
    
    leftprobe = [0 for _ in range(DIGIT_DATUM_WIDTH)]
    rightprobe = [0 for _ in range(DIGIT_DATUM_WIDTH)]
    topprobe = [0 for _ in range(DIGIT_DATUM_HEIGHT)]
    bottomprobe = [0 for _ in range(DIGIT_DATUM_HEIGHT)]

    for x in range(DIGIT_DATUM_WIDTH):
      bestsofar = 0
      currentline = 0
      transitions = 0
      last_pixel = None
      blocktransitions = 0
      lastblock = None
      firstblack = None
      lastblack = None
      for y in range(DIGIT_DATUM_HEIGHT):
          if datum.getPixel(x, y) > 0:
              features[(x,y)] = 1
              averagey += y
              currentline += 1
              transitions += 1 if last_pixel == 1 else 0
              last_pixel = 0
              if firstblack == None:
                firstblack = y
              lastblack = y 
              totalblack += 1 
          else:
              features[(x,y)] = 0
              bestsofar = max(bestsofar, currentline)
              currentline = 0
              transitions += 1 if last_pixel == 0 else 0
              last_pixel = 1  
          
          if x < DIGIT_DATUM_WIDTH - 1 and y < DIGIT_DATUM_HEIGHT - 1:
            darkness = 0.0
            for xp in range(2):
              for yp in range(2):
                darkness += 1 if datum.getPixel(x + xp, y + yp) > 0 else 0
            features["block2_" + str(x) + "," + str(y)] = darkness 
            blocktransitions += 1 if darkness == lastblock else 0
            lastblock = darkness

      bottomprobe[x] = DIGIT_DATUM_HEIGHT if firstblack == None else lastblack
      topprobe[x] = 0 if firstblack == None else firstblack      

      features["height"+ str(x)] = 0 if firstblack == None else lastblack - firstblack
      features["transitions"+ str(x)] = transitions
      features["maxcol_" + str(x)] = bestsofar 
    
    for y in range(DIGIT_DATUM_HEIGHT):
      bestsofar = 0
      currentline = 0
      transitions = 0
      last_pixel = None
      firstblack = None
      lastblack = None
      for x in range(DIGIT_DATUM_WIDTH):
          if datum.getPixel(x, y) > 0:
              averagex += x
              currentline += 1
              transitions += 1 if last_pixel == 1 else 0
              last_pixel = 0
              if firstblack == None:
                firstblack = x
              lastblack = x  
          else:
              bestsofar = max(bestsofar, currentline)
              currentline = 0
              transitions += 1 if last_pixel == 0 else 0
              last_pixel = 1
              
      features["width"+ str(y)] = 0 if firstblack == None else lastblack - firstblack
      features["transitions"+ str(y)] = transitions 
      features["maxrow_" + str(y)] = bestsofar
      minthickness = min(bestsofar, minthickness)
      maxthickness = max(bestsofar, maxthickness)
      leftprobe[y] = DIGIT_DATUM_WIDTH if firstblack == None else firstblack
      rightprobe[y] = 0 if firstblack == None else lastblack

 
    features["strokediff"] = (maxthickness - minthickness)
    features["strokediff2"] = (maxthickness - minthickness) ** 2
 
    minx = float("inf")
    miny = float("inf")
    maxx = float("-inf")
    maxy = float("-inf")
    for x in range(DIGIT_DATUM_WIDTH):
      for y in range(DIGIT_DATUM_HEIGHT):
        if datum.getPixel(x,y) > 0:
          minx = min(minx, x)
          miny = min(miny, y)
          maxx = max(maxx, x)
          maxy = max(maxy, y)         
    
    totalwidth = maxx - minx
    totalheight = maxy - miny     
    
    bottom = 0.0
    for x in range(minx, maxx + 1):
      for y in range(miny, (miny + maxy)/2):
        if datum.getPixel(x,y) > 0:
          bottom += 1 
          
    left = 0.0
    for x in range(minx, (miny + maxx)/2):
      for y in range(miny, maxy + 1):
        if datum.getPixel(x,y) > 0:
          left += 1 

    """
    features["totalheight"] = totalheight
    features["totalwidth"] = totalwidth
    features["blackinbottom"] = bottom / totalblack
    features["blackinleft"] = left / totalblack

    features["shiftvert"] = averagey - (maxy + miny)/2
    features["shifthorz"] = averagex - (maxx + minx)/2
    features["totalblack"] = totalblack
    """

    totalwhite = DIGIT_DATUM_HEIGHT * DIGIT_DATUM_WIDTH - totalblack
      
    hasLoops = 1 if 1.0 * whitespace.totalCount()/totalwhite < 0.98 else 0    
    
    #if hasLoops == 1:
    #  print datum
    
    features["hasLoops"] = hasLoops
    
    
    #count peaks
    leftpeaks = 0
    rightpeaks = 0
    toppeaks = 0
    bottompeaks = 0

    breadth = 2
    for x in range(breadth, DIGIT_DATUM_WIDTH - breadth):
      features["leftprobe" + str(x)] = leftprobe[x]
      features["rightprobe" + str(x)] = rightprobe[x]

      leftpeaks += 1 if leftprobe[x] > leftprobe[x-breadth] and leftprobe[x] > leftprobe[x+breadth] else 0
      rightpeaks += 1 if rightprobe[x] < rightprobe[x-breadth] and rightprobe[x] < rightprobe[x+breadth] else 0

    for y in range(breadth, DIGIT_DATUM_HEIGHT - breadth):
      features["topprobe" + str(y)] = topprobe[y]
      features["bottomprobe" + str(y)] = bottomprobe[y]

      toppeaks += 1 if topprobe[y] < topprobe[y-breadth] and topprobe[y] < topprobe[y+breadth] else 0
      bottompeaks += 1 if bottomprobe[y] > bottomprobe[y-breadth] and bottomprobe[y] > bottomprobe[y+breadth] else 0

    features["leftpeaks"] = leftpeaks
    features["rightpeaks"] = rightpeaks
    features["toppeaks"] = toppeaks
    features["bottompeaks"] = bottompeaks

    """
    if leftpeaks > 0:
      print leftpeaks
      print datum
    
    print "COMPARE"
    print datum
    for i in range(DIGIT_DATUM_HEIGHT):
      print "#" * leftprobe[i]

    for i in range(DIGIT_DATUM_HEIGHT):
      print " " * rightprobe[i] + "#" * (DIGIT_DATUM_HEIGHT - rightprobe[i])
    """


    #features["blackinfirstcolband"] = 1.0 * blackinfirstcolband / totalblack
    #features["blackinmidcolband"] = 1.0 * blackinmidcolband / totalblack
    #features["blackinlastcolband"] = 1.0 * blackinlastcolband / totalblack
    #features["averagey"] = averagey / totalblack
    #features["averagex"] = averagex / totalblack
    features["aspectratio"] = (1.0 * totalheight) / totalwidth


    """
    width = DIGIT_DATUM_WIDTH/5
    height = DIGIT_DATUM_HEIGHT/5
    for _x in range(width):
      for _y in range(height):
        x = _x * 5
        y = _y * 5
        count = 0.0
        for i in range(width):
          for j in range(height):
            count += 1 if datum.getPixel(x + i, y + j) > 0 else 0
        features["concentration" + str(x) + "," + str(y)] = count / totalblack
    

    
    for _x in range(DIGIT_DATUM_WIDTH - 6):
      for _y in range(DIGIT_DATUM_HEIGHT - 6):
        grid = [[0,0,0]] * 3
        for x in range(3):
          for y in range(3):
            ul = 1 if datum.getPixel(_x + x * 2, _y + y * 2) > 0 else 0
            ur = 1 if datum.getPixel(_x + x * 2 + 1, _y + y * 2) > 0 else 0
            ll = 1 if datum.getPixel(_x + x * 2, _y + y * 2 + 1) > 0 else 0
            lr = 1 if datum.getPixel(_x + x * 2 + 1, _y + y * 2 + 1) > 0 else 0
            grid[x][y] = 1 if ul + ur + ll + lr > 2 else 0
            
        features["vertgrade" + str(x) + "," + str(y)] = 1 if (grid[0][0] + grid[1][0] + grid[2][0] == 3 or 
                                                              grid[0][1] + grid[1][1] + grid[2][1] == 3 or 
                                                              grid[0][2] + grid[1][2] + grid[2][2] == 3 ) else 0
        features["horzgrade" + str(x) + "," + str(y)] = 1 if (grid[0][0] + grid[0][1] + grid[0][2] == 3 or
                                                              grid[1][0] + grid[1][1] + grid[1][2] == 3 or
                                                              grid[2][0] + grid[2][1] + grid[2][2] == 3 ) else 0
        features["diag1grade" + str(x) + "," + str(y)] = 1 if (grid[0][0] + grid[1][1] + grid[2][2] == 3 and
                                                               grid[2][0] + grid[0][2] == 0) else 0
        features["diag2grade" + str(x) + "," + str(y)] = 1 if (grid[2][0] + grid[1][1] + grid[0][2] == 3 and
                                                               grid[0][0] + grid[2][2] == 0) else 0
        features["corner" + str(x) + "," + str(y)] = 1 if (grid[0][0] + grid[0][2] + grid[2][2] + grid[2][0] == 3) else 0
    """

    return features



def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    "*** YOUR CODE HERE ***"

    currentGameState = state
    currentPos = currentGameState.getPacmanPosition()
    currentCaps = currentGameState.getCapsules() #.asList()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentGhostDistAndTimer = map(lambda x : (search.mazeDistance(x.getPosition(), currentPos, state), x.scaredTimer), currentGhostStates)
    currentFoodDists = map(lambda x : (x, abs(x[0] - currentPos[0]) + abs(x[1] - currentPos[1])), currentFood)
    currentCapDists = map(lambda x : (x, abs(x[0] - currentPos[0]) + abs(x[1] - currentPos[1])), currentCaps)
    currentClosestGhost = min(currentGhostDistAndTimer, key=lambda state: state[0])
    currentScaredGhosts = [x for x in currentGhostDistAndTimer if x[1] > 0]
    currentClosestScaredGhost = None if len(currentScaredGhosts) == 0 else min(currentScaredGhosts, key=lambda state: state[0])


    newGameState = state.generatePacmanSuccessor(action)
    newPos = newGameState.getPacmanPosition()
    newCaps = newGameState.getCapsules() #.asList()
    newFood = newGameState.getFood().asList()
    newGhostStates = newGameState.getGhostStates()
    newGhostDistAndTimer = map(lambda x : (search.mazeDistance(x.getPosition(), newPos, state), x.scaredTimer ), newGhostStates)
    newFoodDists = map(lambda x : (x, abs(x[0] - newPos[0]) + abs(x[1] - newPos[1])), newFood)
    newCapDists = map(lambda x : (x, abs(x[0] - newPos[0]) + abs(x[1] - newPos[1])), newCaps)
    newClosestGhost = min(newGhostDistAndTimer, key=lambda state: state[0])


    #print dir(search)

    """
     closestGhost
     closestGhostTimer
     closestFood
     score
     closestCapsule
    """

    #print closestGhost
    #features["foodCount"] = len(foodDists)
    features["closestScaredGhostDist"] =  0 if currentClosestScaredGhost == None else currentClosestScaredGhost[0] ** 2
    features["closestGhostDist"] = 0 if newClosestGhost[0] == 0 else 1 / newClosestGhost[0] ** 2
    #features["closestGhostTimer"] = newClosestGhost[1]
    features["closestGhostDistChange"] = newClosestGhost[0] - currentClosestGhost[0]
    features["closestGhostTimerChange"] = newClosestGhost[1] - currentClosestGhost[1]
    features["closestFoodDist"] = 0 if len(newFoodDists) == 0 else 1 / search.mazeDistance(min(newFoodDists, key=lambda x: x[1])[0], newPos, state)
    features["closestCapDist"] = 0 if len(newCapDists) == 0 else 1 / search.mazeDistance(min(newCapDists, key=lambda x: x[1])[0], newPos, state)
    features["newScore"] = newGameState.getScore() ** 2
    #features["isWin"] = 1 if newGameState.isWin() else 0
    #features["isLose"] = 1 if newGameState.isLose() else 0
    #features["Stop"] = -1000 if action == "Stop" else 0

    """
    for food in newFood:
      features["food" + str(food)] = 1

    for cap in newCaps:
      features["caps" + str(cap)] = 1

    index = 0
    for ghost in sorted(newGhostDistAndTimer, key=lambda x: x[0]):
      features["ghost" + str(index)] = ghost[0]
      index += 1
    """

  
    #print state
    #print features
  



    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    """
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print "==================================="
            print "Mistake on example %d" % i
            print "Predicted %d; truth is %d" % (prediction, truth)
            print "Image: "
            print rawTestData[i]
    """


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
