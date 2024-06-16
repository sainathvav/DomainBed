# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm ERM \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_ERM_ResNet18 \

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm ERM \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_ERM_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/NICO/NICO_ERM_Teacher.pkl

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm ERM \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_ERM_Teacher \

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm ERM \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_ERM_Student_AUX \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/TerraIncognita/TerraIncognita_ERM_Teacher.pkl

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home/durga/KD/RobustDistillation/Robust\ Distillation/KD/DomainBed/domainbed/data \
#        --algorithm ERM \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHomedum \
#        --model_name OfficeHome_ERM_Teacher_TEST 

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home/durga/KD/RobustDistillation/Robust\ Distillation/KD/DomainBed/domainbed/data \
#        --algorithm ERM \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHomeDum \
#        --model_name OfficeHome_ERM_Student \
#        --KD 1 \
#        --teacher_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/OfficeHomedum/OfficeHome_ERM_Teacher.pkl

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm ERM \
#        --test_env 1  \
#        --dataset DomainNet \
#        --output_dir train_output/DomainNet \
#        --model_name DomainNet_ERM_Teacher \

# CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm ERM \
#        --test_env 1  \
#        --dataset DomainNet \
#        --output_dir train_output/DomainNet \
#        --model_name DomainNet_ERM_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/DomainNet/DomainNet_ERM_Teacher.pkl

CUDA_VISIBLE_DEVICES="2"  python3  DomainBed/domainbed/train.py \
       --data_dir=/home/durga/KD/RobustDistillation/Robust\ Distillation/KD/DomainBed/domainbed/data \
       --algorithm MTL \
       --test_env 3 \
       --dataset PACS \
       --EE 0 \
       --KD 1 \
       --base_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/PACS/PACS_MTL_student.pkl \
       --output_dir train_output/PACS \
       --model_name PACS_MTL_student \
       --teacher_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/PACS/PACS_MTL_Teacher.pkl