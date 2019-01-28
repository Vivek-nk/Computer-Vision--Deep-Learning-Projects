# Computer Vision & Image Processing:  Image Annotation Project

## Image Annotation for Computer Vision

## Improve machine learning object recognition with quality training data


## Background
Computer vision solutions are becoming accessible to almost every industry, from autonomous vehicles and medical research to retail and agriculture, all thanks to dramatic advances in machine learning scalability and open-source modeling approaches. Attempted deployments of these models oftentimes fail accuracy standards because of the same fundamental problem: inadequate case-specific training data at scale. 

Bounding Boxes- Detect areas that correspond to objects, such as cars or pedestrians, in varied settings.

Polygons- Outline the shape of an object, such as the pixel-area of abnormal cells.

Dots- Define the pixel coordinate of product inventory, or other points of reference

ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. ImageNet is useful resource for researchers, educators, students. 

http://image-net.org


The project helps Colorado Parks and Wildlife (CPW) perform automatic annotation of animal tracking.

There are 45 animal categories to determine within areas that CPW 
  Bobcat	Badger
  
  Mountain Lion	Butterfly
  
  Coyote	Hawk
  
  Elk	Magpie
  
  Ermine	Mule
  
  Human	Prairie_dog
  
  Lynx	Pronghorn
  
  Marten	Rabbit
  
  Moose	Raven
  
  Mule Deer	Sage_grouse
  
  Porcupine	Striped_skunk
  
  Ptarmigan	Cow
  
  Red Fox	Horse
  
  Red Squirrel	Domestic Sheep
  
  Snowshoe Hare	Turkey
  
  Bear	Gray Jay
  
  Bird	Dusky grouse
  
  Mountain Goat	Mouse
  
  Marmot	Bat
  
  Bighorn Sheep	Striped Skunk
  
  Raccoon	House Wren
  
  Dog	Stellar's Jay
  
  Chipmunk	Dark-eyed Junco
  

In general, we can annotate using BoundingBox in a Jupyter notebook to generate data for performing deep learning image classifiers and object detectors.

Step: Run each cell of the AnnotateJupyter notebook to initialize the system

After running the final cell:


  anotateImage = AnnotationProject()
  
  annotateImage.runAnnotator()
  
 The system will show an image with 4 buttons: 
 
      Button	            Description
      
      Clear	              Clear all bounding boxes stored for the image
      
      Show	              Show all bounding boxes (coordinates) 
      
      RejectImage	        Used to determine that image is not going to be effective for training a deep learning model. Examples: no animal, major blockage of viewing animal (fence in front of animal).
      
      NextImage	          Store the image bounding boxes and then move on to the next image to annotate

Use your mouse to select a bounding box over the image.


