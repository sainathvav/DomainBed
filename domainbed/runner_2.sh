# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_MTL_Teacher \

# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_MTL_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/NICO/NICO_MTL_Teacher.pkl

# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_MTL_Teacher \

# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_MTL_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/TerraIncognita/TerraIncognita_MTL_Teacher.pkl

# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHome \
#        --model_name OfficeHome_MTL_Teacher \

# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHome \
#        --model_name OfficeHome_MTL_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/OfficeHome/OfficeHome_MTL_Teacher.pkl


# CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm MTL \
#        --test_env 1  \
#        --dataset DomainNet \
#        --output_dir train_output/DomainNet \
#        --model_name DomainNet_MTL_Teacher \

CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
       --data_dir=/home/durga/KD/RobustDistillation/Robust\ Distillation/KD/DomainBed/domainbed/data \
       --algorithm ERM \
       --test_env 3 \
       --dataset VLCS \
       --EE 1 \
       --KD 0 \
       --base_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/VLCS/VLCS_ERM_forecast_student.pkl \
       --output_dir train_output/VLCS \
       --model_name VLCS_ERM_forecast_student \
       --teacher_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/VLCS/VLCS_ERM_Teacher.pkl