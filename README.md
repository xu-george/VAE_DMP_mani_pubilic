# Using dmp_cVAE methods to learning mulit-task learning with only one demonstration 

## 1. Introduction 
We build a pytorch version DMP(dynamic motion primitive) and cVAE(conditional variational autoencoder) to learn the multi-task learning with only one demonstration. Below function are provided in this repository:

1. Provide a pytorch version DMP, weight can be trained or calcualted as classic DMP.
2. Fine-tune cVAE-DMP to pass via-point.
3. Test on number writing dataset.
4. Test on the robotic grasping, pushing, reaching on ur10 in the pybullet environment.

## 2. Installation
1. create a virtual environment with miniconda
```bash
conda create -n vae_dmp python=3.8
```

2. Clone the repository to your local machine
```bash
git clone https://github.com/xu-george/VAE_DMP_mani_pubilic
```

3. install the required packages in the requirements.txt
```bash
pip install -r requirements.txt
```
## 3. number writing dataset usage
### 1. create data set
we collect 10 digit dataset in the number writing dataset with data augmentation. The dataset is saved in the .data/number_write folder. You can
create a dataset using the code in ./number_write_task/create_pytorch_data_set.ipynb

### 2. train VAE-DMP
in number_write_task folder, you can train the vae-dmp model using the following command
```bash
python train_torque_vae.ipynb
```
### 3. test VAE-DMP 
in number_write_task folder, you can test the vae-dmp model using the following command to check the endpoint and via-point constrain
```bash
python End_point_test.ipynb
```
## 4. robotic task usage
### 1. create the data set
we collect one demonstration for each task in the robotic task. The dataset is saved in the .data/manipulation_task folder. 

You can create a dataset using the code in ./manipulation_task/create_manipulation_data_set.ipynb

### 2. train VAE-DMP
in manipulation_task folder, you can train the vae-dmp model using the following command
```bash
python train_torque_vae.ipynb
```
### 3. test VAE-DMP
in manipulation_task folder, you can test the vae-dmp model using the following command to check the three tasks
```bash
python test_grasping.ipynb
python test_pushing.ipynb
python test_reaching.ipynb
```