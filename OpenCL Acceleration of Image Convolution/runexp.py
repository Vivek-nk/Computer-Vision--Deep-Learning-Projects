#!/usr/bin/python

import sys
import os
import subprocess
import re 
import time


file = open("Execution_Time.csv", 'w');

def runtest(Img_Width, Img_Height, Filter_Size):
    compileCmd = "cc"
    compileCmd = compileCmd + " -DIMAGE_W=" + str(Img_Width)
    compileCmd = compileCmd + " -DIMAGE_H=" + str(Img_Height)
    compileCmd = compileCmd + " -DFILTER_W=" + str(Filter_Size)
    compileCmd = compileCmd + " -Wall -Wno-comment -Wno-deprecated-declarations -O2 -std=gnu99 -lOpenCL -lm convolveCL.c -o convolveCL"
    


    # Compile the Code 
    os.system(compileCmd)
    
    # Run the Code 
    start_timer = time.time()
    runCmd = "./convolveCL"
    os.system(runCmd) 
    end_timer = time.time()

    #The following captures the output unlike the os.system control
    proc = subprocess.Popen(runCmd, shell=True, bufsize=256, stdout=subprocess.PIPE)
    for line in proc.stdout:
        temp =  str(end_timer - start_timer) + ',' + str(Img_Width) + ',' + str(Img_Height)+ ',' + str(Filter_Size) + '\n'
    file.write(temp)
    


# Running Convolve.CL Code for fictional images of size: 512x512, 1024x1024, 2048x2048 (Image Width X Image Height)
# For Filter Size: 3x3, 5x5, 7x7, 9x9, 11x11, 13x13
for Img_Width in [512,1024,2048]:
    for Img_Height in [512,1024,2048]:
      	for Filter_Size in [3,5,7,9,11,13]:
            runtest(Img_Width, Img_Height, Filter_Size)

