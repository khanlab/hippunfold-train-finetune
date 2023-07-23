
configfile: 'config.yml'

wildcard_constraints:
    app='hippunfold',
    hemi='L|R',
    subject='[a-zA-Z0-9]+',
    i='[0-9]+'

def get_zip_file(wildcards):
    """ assuming we have zipfiles named as:  sub-{subject}_<...>.zip, 
        e.g.  diffparc.zip 
    """
    return config['in_zip'][wildcards.app]


(subjects,hemis)= glob_wildcards(config['hippunfold_lbl'])

testing_subjects=[]
training_subjects=subjects



localrules: cp_training_img,cp_training_lbl,plan_preprocess,create_dataset_json


rule all_train:
    input:
       expand('trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}.round_{i}.DONE',fold=range(5), arch=config['architecture'], dataset=config['dataset'], trainer=config['trainer'],plans=config['plans'],i=16)

 
rule all_model_tar:
    """Target rule to package trained model into a tar file"""
    input:
        model_tar = expand('trained_model.{arch}.{dataset}.{trainer}.tar',arch=config['architecture'], dataset=config['dataset'], trainer=config['trainer'],plans=config['plans'])


rule all_predict:
    input:
        testing_imgs = expand('raw_data/nnUNet_predictions/{dataset}/{trainer}__{plans}__{arch}/hcp_{subject}{hemi}.nii.gz',subject=testing_subjects, hemi=hemis, arch=config['architecture'], dataset=config['dataset'], trainer=config['trainer'],plans=config['plans'],allow_missing=True),
 

rule get_from_zip:
    """ This is a generic rule to make any file within the {app} subfolder, 
        by unzipping it from a corresponding zip file"""
    input:
        zip=get_zip_file
    output:
        '{app}/{file}' # you could add temp() around this to extract on the fly and not store it
    shell:
        'unzip -d {wildcards.app} {input.zip} {wildcards.file}'

 

rule cp_training_img:
    input: 
        nii = 'hippunfold/work/sub-{subject}/anat/sub-{subject}_hemi-{hemi}_space-corobl_desc-preproc_T1w.nii.gz',
    output: 'raw_data/{dataset}/imagesTr/dhcp_{subject}{hemi}_0000.nii.gz'
    threads: 32 #to make it serial on a node
    group: 'preproc'
    shell: 'cp {input} {output}'

rule cp_testing_img:
    input: 
        nii = 'hippunfold/work/sub-{subject}/anat/sub-{subject}_hemi-{hemi}_space-corobl_desc-preproc_T1w.nii.gz',
    output: 'raw_data/{dataset}/imagesTs/dhcp_{subject}{hemi}_0000.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'cp {input} {output}'


rule cp_training_lbl:
    input:
        nii = config['hippunfold_lbl']
    output: 'raw_data/{dataset}/labelsTr/dhcp_{subject}{hemi}.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'cp {input} {output}'


rule create_dataset_json:
    input: 
        training_imgs = expand('raw_data/{dataset}/imagesTr/dhcp_{subject}{hemi}_0000.nii.gz',zip,subject=training_subjects, hemi=hemis,allow_missing=True),
        training_lbls = expand('raw_data/{dataset}/labelsTr/dhcp_{subject}{hemi}.nii.gz',zip,subject=training_subjects, hemi=hemis,allow_missing=True),
    params:
        training_imgs_nosuffix = expand('raw_data/{dataset}/imagesTr/dhcp_{subject}{hemi}.nii.gz',zip,subject=training_subjects, hemi=hemis,allow_missing=True),
    output: 
        dataset_json = 'raw_data/{dataset}/dataset.json'
    group: 'preproc'
    script: 'create_json.py' 
    
def get_nnunet_env(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env'].items()])
 
def get_nnunet_env_tmp(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env_tmp'].items()])
 
rule plan_preprocess:
    input: 
        dataset_json = 'raw_data/{dataset}/dataset.json'
    params:
        nnunet_env_cmd = get_nnunet_env,
        dataset_id = lambda wildcards: re.search('([0-9]+)\w*',wildcards.dataset).group(1),
    output: 
        dataset_json = 'preprocessed/{dataset}/dataset.json'
    group: 'preproc'
    resources:
        threads = 8,
        mem_mb = 16000
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNetv2_plan_and_preprocess  -d {params.dataset_id} --verify_dataset_integrity'

rule train_fold_init_round:
    input:
        dataset_json = 'preprocessed/{dataset}/dataset.json',
    params:
        nnunet_env_cmd = get_nnunet_env_tmp,
        rsync_to_tmp = f"rsync -av {config['nnunet_env']['nnUNet_preprocessed']} $TMPDIR",
        output_dir = 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}',
    output:
        training_done = 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}.round_0.DONE'
    threads: 16
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 180,
    shell:
        '{params.nnunet_env_cmd} && '
        '{params.rsync_to_tmp} && '
        'touch {output} && '
        'set +e; '
        'nnUNetv2_train  {wildcards.dataset} {wildcards.arch} {wildcards.fold}'


     
rule train_fold_round_i:
    input:
        dataset_json = 'preprocessed/{dataset}/dataset.json',
        training_done = lambda wildcards: 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}.round_{i}.DONE'.format(dataset=wildcards.dataset,trainer=wildcards.trainer,plans=wildcards.plans,arch=wildcards.arch, fold=wildcards.fold, i=int(wildcards.i)-1)
    params:
        nnunet_env_cmd = get_nnunet_env_tmp,
        rsync_to_tmp = f"rsync -av {config['nnunet_env']['nnUNet_preprocessed']} $TMPDIR",
        output_dir = 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}',
    output:
        training_done = 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}.round_{i}.DONE'
#        latest_model = 'trained_models/nnUNet/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}/checkpoint_best.pth',
#        best_model = 'trained_models/nnUNet/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}/checkpoint_final.pth'
    threads: 16
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 180,
    shell:
        '{params.nnunet_env_cmd} && '
        '{params.rsync_to_tmp} && '
        'touch {output} && '
        'set +e; '
        'nnUNetv2_train --c {wildcards.dataset} {wildcards.arch} {wildcards.fold}'


rule package_trained_model:
    """ Creates tar file for performing inference with workflow_inference -- note, if you do not run training to completion (1000 epochs), then you will need to clear the snakemake metadata before running this rule, else snakemake will not believe that the model has completed. """
    input:
        training_done = 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}.DONE'

    params:
        trained_model_dir = config['nnunet_env']['nnUNet_results'],
        files_to_tar = 'nnUNet/{dataset}/{trainer}__{plans}__{arch}'
#        latest_model = expand('trained_models/nnUNet/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}/{checkpoint}.pth',fold=range(5),allow_missing=True),
#        latest_model_pkl = expand('trained_models/nnUNet/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}/{checkpoint}.pth.pkl',fold=range(5),allow_missing=True),

    output:
        model_tar = 'trained_model.{arch}.{dataset}.{trainer}.tar'
    shell:
        'tar -cvf {output} -C {params.trained_model_dir} {params.files_to_tar}'


rule predict_test_subj:
    input:
        training_done = 'trained_models/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}.DONE',
#        in_training_folder = expand('trained_models/nnUNet/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}',fold=range(5),allow_missing=True),
#        latest_model = expand('trained_models/nnUNet/{dataset}/{trainer}__{plans}__{arch}/fold_{fold}/{checkpoint}.pth',fold=range(5),allow_missing=True),
        testing_imgs = expand('raw_data/{dataset}/imagesTs/hcp_{subject}{hemi}_0000.nii.gz',subject=testing_subjects, hemi=hemis, allow_missing=True),
    params:
        in_folder = 'raw_data/{dataset}/imagesTs',
        out_folder = 'raw_data/nnUNet_predictions/{dataset}/{trainer}__{plans}__{arch}',
        nnunet_env_cmd = get_nnunet_env,
        checkpoint = 'checkpoint_final',
    output:
        testing_imgs = expand('raw_data/nnUNet_predictions/{dataset}/{trainer}__{plans}__{arch}/{checkpoint}/hcp_{subject}{hemi}.nii.gz',subject=testing_subjects, hemi=hemis, allow_missing=True),
    threads: 8 
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 30,
    group: 'inference'
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNetv2_predict  -chk {params.checkpoint}  -i {params.in_folder} -o {params.out_folder} -t {wildcards.task}'

   
        

