# AEGeAN
Deeper DCGAN with AE stabilization

Parallel training of generative adversarial network as an autoencoder with dedicated losses for each stage. Generator class has conditional `.forward()` method for enhanced ergonomics. Autoencoding pass seems to avoid mode collapse and recover faster if Generator is not doing well.

Has been used successfully with as few as ~200 images in the source folder.

Builds on the DCGAN PyTorch demo. This one generates images upto 1024x1024 so it can use a lot of VRAM.

Should work when the "dataroot" is configured ImageNet style: ".../a_dir_of_images/what_would_be_a_label" or ".../cat_pics/cute_cats/cat_001.jpg"

Have fun!

Examples of generated drawings [here](http://www.aiartonline.com/design/318/)

A description of the project for which this was developed can be found [here](https://tylerkvochick.com/work/gen-the-gen).
