# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_Mixup_Teacher \

# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 0  \
#        --dataset NICO \
#        --output_dir train_output/NICO \
#        --model_name NICO_Mixup_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/NICO/NICO_Mixup_Teacher.pkl

# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_Mixup_Teacher \

# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 2  \
#        --dataset TerraIncognita \
#        --output_dir train_output/TerraIncognita \
#        --model_name TerraIncognita_Mixup_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/TerraIncognita/TerraIncognita_Mixup_Teacher.pkl

# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHome \
#        --model_name OfficeHome_Mixup_Teacher \

# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 3  \
#        --dataset OfficeHome \
#        --output_dir train_output/OfficeHome \
#        --model_name OfficeHome_Mixup_Student \
#        --KD 1 \
#        --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/OfficeHome/OfficeHome_Mixup_Teacher.pkl

# CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
#        --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
#        --algorithm Mixup \
#        --test_env 1  \
#        --dataset DomainNet \
#        --output_dir train_output/DomainNet \
#        --model_name DomainNet_Mixup_Teacher \

CUDA_VISIBLE_DEVICES="3"  python3  DomainBed/domainbed/train.py \
       --data_dir=/home1/durga/sainath/Robust\ Distillation/KD/DomainBed/domainbed/data/ \
       --algorithm Mixup  \
       --test_env 1  \
       --dataset DomainNet \
       --output_dir train_output/DomainNet \
       --model_name DomainNet_Mixup_Student \
       --KD 1 \
       --teacher_model_path /home1/durga/sainath/Robust\ Distillation/KD/train_output/DomainNet/DomainNet_Mixup_Teacher.pkl