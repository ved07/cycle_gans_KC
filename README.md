# CycleGANs King's Certificate Project
This project is my King's Certificate Research Project submitted in 2022, although the project was a collaborative effort, a majority of the write-up (linked at the bottom of this ReadMe) and the all the code, is my own creation.
## Objective
To take images from a simulated environment, and convert them into realistic training data, ideally using GANs.
## Datasets used
Used the Kitti and Carla datasets, I believe this is why the final results were not as desirable as hoped. 
## Implementations
Note that the repository contains one extra implementation, this did not yield any significant results so it was not included in the ReadMe.
### Implementation 1
Used a simple GAN with the below architecture. Did not perform Style Transfer, just attempted to generate realistic images.The model consisted of transposed-convolution layers, condensing layers, and a single dense layer. 
![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/b687f153-2d54-4856-b6bb-fa416b947e87) ![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/ce1b43de-8634-426e-a2c5-33817e48a677)

It was obvious that the model did not have sufficient parameters to learn, and that the discriminator out-performed the generator.

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/6496b416-3003-4254-92bd-0315487f9c8b)

The results produced noisy checkerboard patterns.

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/8328a067-60b8-437a-b9d3-e6d0eb4982d7)

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/a7bfabb5-3eec-43ca-bb9a-46545738070e)


### Implementation 2
Used a CycleGAN that followed the below architecture.

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/ae75cb4f-59f8-4aa9-8f8e-2b878eec8372)

This did perform slightly better than the GAN, but it was evident that there was a lot of instability _potentially due to the identity loss component, and batch normalisation._

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/dcbd4a76-df30-4b2d-a36a-961e58346b5c)

### Implementation 3
Removed the batch norm and identity components of the architecture and saw a greater "acknowledgement" of the key features within the image.

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/2c17bbac-577e-4699-a850-ad3a559261e9)

However, the model did not really perform style transfer. So I changed the architecture to the final implementation.

### Implementation 4

This was the final implementation, and it continued to suffer from a lot of the problems that were evident through the duration of the project, specifically the discriminator massively overperforming the generator.

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/33026bfe-83c5-4429-8ff5-e05355d3d4b0)

However, the results did have some of the key colours and features present in the realistic (Kitti) training data, with it being somewhat adept at matching trees from the input domain to the output domain. It is also able to generate a realistic looking sky and lighting.

![image](https://github.com/ved07/cycle_gans_KC/assets/49959052/e7bf88ee-f5cf-40d8-8975-c82c151dd5ce)

## Final Remarks
The final results show some sembelance of the output domain, however, it is evident that key features are not being "understood" by the model, for example cars, buildings and roads. Looking into the subset of the Kitti dataset selected, it is noticeable that the data does not contain many urban-esque images, whereas Carla is situated entirely within an urban environment.  

## Link to our paper: https://drive.google.com/file/d/1ntK0YxXOk12dFYaksXYfsR051rMu1iZi/view
