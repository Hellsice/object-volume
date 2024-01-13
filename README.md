# object-volume
This is a project i did for my Physics study.
I used opencv and a chessboard pattern to calibrate my phone, which i used to take a series of images from different angles of 3 objects.
Colmap was used to find the camera poses of all images, of which the results are saved in "colmap reconstruction".
The images taken are in folders "images" and "volume calibration".
Note, i made the mistake of using a too high resolution. So quite a lot of RAM is used for these images.

The space carving code comes from zinsmatt/SpaceCarving.
