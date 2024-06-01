
# Factor for downscaling of test images
SCALE = 0.5
#Specify image paths
img_path1 = r'/home/drdrew/Projects/diplom/semi-global-matching/Code/left.png'
#left.png'
#im2.png'
img_path2 = r'/home/drdrew/Projects/diplom/semi-global-matching/Code/right.png'
#right.png'
#im6.png'


###################
# CAMERA PARAMS
###################
# The focal length of the two cameras, taken from calib.txt
FOCAL_LENGTH = 3979.911
# The distance between the two cameras, taken from calib.txt
X_A=1244.772
X_B=1369.115
Y= 1019.507
DOFFS = X_B-X_A
CAMERA_DISTANCE = 193.001
