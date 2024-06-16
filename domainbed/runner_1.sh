# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_CORAL_Teacher \

# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_CORAL_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/NICO/NICO_CORAL_Teacher.pkl

# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_CORAL_Teacher \

# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_CORAL_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/TerraIncognita/TerraIncognita_CORAL_Teacher.pkl

# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHome \
#        --model_name OfficeHome_CORAL_Teacher \

# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHome \
#        --model_name OfficeHome_CORAL_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/OfficeHome/OfficeHome_CORAL_Teacher.pkl

# CUDA_VISIBLE_DEVICES="1"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm CORAL \
#        --test_env 1  \
#        --dataset DomainNet \
#        --output_dir train_output/DomainNet \
#        --model_name DomainNet_CORAL_Teacher \

CUDA_VISIBLE_DEVICES="0"  python3  DomainBed/domainbed/train.py \
       --data_dir=/home/durga/KD/RobustDistillation/Robust\ Distillation/KD/DomainBed/domainbed/data \
       --algorithm ERM \
       --test_env 3 \
       --dataset DomainNet \ 
       --EE 0 \
       --base_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/domain_net/domain_net_ERM_Teacher.pkl \
       --output_dir train_output/domain_net \
       --model_name domain_net_ERM_Teacher 
#      --teacher_model_path /home/durga/KD/RobustDistillation/Robust\ Distillation/KD/train_output/domain_net/domain_net_ERM_Teacher.pkl