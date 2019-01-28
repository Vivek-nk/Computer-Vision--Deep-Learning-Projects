import os

def Run_Job(command, inputFile, networkFile, outputFile):
    # Run the code
    runCmd =  " ".join([command, inputFile,networkFile,outputFile])
    print runCmd 
    os.system(runCmd)

def Process_Parms(inputList, networkList):
    runList = [(filter(lambda x: networkFile.strip('.net').split('_')[1][1:]+'.in' in x, inputList)[0],networkFile) for networkFile in networkList]
    return runList

inputDir = 'Inputs'
networkDir = 'Networks'
outputFile = 'Results.txt'
command = './a.out'

# Get list of input files
inputList = [os.path.join(inputDir,file) for file in os.listdir(inputDir) if file.endswith('.in')]

# Get list of network files
networkList = [os.path.join(networkDir,file) for file in os.listdir(networkDir) if file.endswith('.net')]


try:
    os.remove(outputFile)
except OSError:
    pass

# Create the pair of input files and networks to run
runList = Process_Parms(inputList, networkList)

# Run the jobs
for (inputFile,networkFile) in runList:
    Run_Job(command,inputFile,networkFile,outputFile)
    
