# MeganeX8K_Distortion_Profiles
A repository of custom distortion profiles for the MeganeX 8K VR headset.

These json files are for use with the CustomHeadset GUI included in the new driver package for the MeganeX Superlight 8K VR headset.  
Each file is a distortion profile that corrects warping and chromatic aberration seen in the image.  
Put the profiles you want to use in the %APPDATA%\CustomHeadset\Distortion folder, then select one in the GUI app.

The MeganeX8K Default.json and MeganeX8K Original.json files are included for reference.
If no file is in the Distortion folder, the default values are used (these are hardcoded in the driver).

More infomation is at https://github.com/sboys3/CustomHeadsetOpenVR

The "distortions" section in the profile is the scaling applied to the image at the specified degree radially from the center.
The first number is degrees, and the second number is percentage.

The "distortionsRed" and "distortionsBlue" sections are for adjusting chromatic aberrations in the red and green channels.
The first number is degrees and the second number is the adjustment factor.

The "smoothAmount" section is the smoothing parameter for the RadialBezier function and its advised to leave it at 0.66
