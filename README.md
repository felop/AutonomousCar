# AutonomousCar

![](https://img.shields.io/tokei/lines/github/felop/autonomouscar)   ![](https://img.shields.io/github/last-commit/felop/autonomouscar)   [![Codacy Badge](https://app.codacy.com/project/badge/Grade/ca4a931bbbc2400cb4a401179d4df188)](https://www.codacy.com/gh/felop/AutonomousCar/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=felop/AutonomousCar&amp;utm_campaign=Badge_Grade)

I built this car for the [IronCar](https://twitter.com/ironcarfrance) race, where all cars are driven by artificial intelligence. It's the result of several weeks of tweaking and programming.
<br/>
<img src="car_pics/IMG-6800.JPG" width=50% align="right">

## ⚙️ hardware : 
* RaspberryPi 3b+
* RC car (T2M Pirate Shooter)
* PiCamera (wide angle)
* RaspberryPi PWM shield
* some voltage regulators
* wifi dongle
* 2x batteries
* relay module as an emergency stop
<br/><br/><br/><br/>

<img src="car_pics/IMG-6805.jpg" width=27% align="left">

## 💿 software :
The car's on-board computer is in charge of processing the images in order to predict the wheels' orientation to correct the car's trajectory. 
This analysis is done by a combination of convolutional and sequential neural networks. These networks predict a direction according to the input image, processed by algorithms in charge of transforming the camera feed into data adapted for the network.

The other on-board computer (it was an Arduino at first but it has been replaced by a PWM shield) is in charge of retrieving the predictions of the network and transmitting them to the different motors.
<br/><br/><br/><br/>
<br/><br/><br/><br/>

<img src="car_pics/IMG-6779.JPG" width=50% align="right">

## 🎛 new upgrades :
To improve stability and thus image quality, I 3D printed two mounts to hold the plexiglass base. 
