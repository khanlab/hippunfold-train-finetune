import pandas as pd




configfile: 'config.yml'


df = pd.read_table(config['training_subjects'])
subjects = list(df.subjects)
training_subjects = subjects
testing_subjects = []


hemis = config['hemis']

print(f'number of training subjects: {len(training_subjects)}')
print(f'number of test subjects: {len(testing_subjects)}')

localrules: cp_training_img,cp_training_lbl,plan_preprocess,create_dataset_json


rule all_train:
    input:
       expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',fold=range(5), arch=config['architecture'], task=config['task'],checkpoint=config['checkpoint'], trainer=config['trainer'])

 
rule all_model_tar:
    """Target rule to package trained model into a tar file"""
    input:
        model_tar = expand('trained_model.{arch}.{task}.{trainer}.{checkpoint}.tar',arch=config['architecture'], task=config['task'], trainer=config['trainer'],checkpoint=config['checkpoint'])


rule all_predict:
    input:
        testing_imgs = expand('raw_data/nnUNet_predictions/{arch}/{task}/{trainer}__nnUNetPlansv2.1/{checkpoint}/hcp_{subject}{hemi}.nii.gz',subject=testing_subjects, hemi=hemis, arch=config['architecture'], task=config['task'], trainer=config['trainer'],checkpoint=config['checkpoint'],allow_missing=True),
 

rule cp_training_img:
    input: 
        nii = config['hippunfold_img']
    output: 'raw_data/nnUNet_raw_data/{task}/imagesTr/hcp_{subject}{hemi}_0000.nii.gz'
    threads: 32 #to make it serial on a node
    group: 'preproc'
    shell: 'cp {input} {output}'

rule cp_testing_img:
    input: 
        nii = config['hippunfold_img']
    output: 'raw_data/nnUNet_raw_data/{task}/imagesTs/hcp_{subject}{hemi}_0000.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'cp {input} {output}'


rule cp_training_lbl:
    input:
        nii = config['hippunfold_lbl']
    output: 'raw_data/nnUNet_raw_data/{task}/labelsTr/hcp_{subject}{hemi}.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'cp {input} {output}'


rule create_dataset_json:
    input: 
        training_imgs = expand('raw_data/nnUNet_raw_data/{task}/imagesTr/hcp_{subject}{hemi}_0000.nii.gz',subject=training_subjects, hemi=hemis, allow_missing=True),
        training_lbls = expand('raw_data/nnUNet_raw_data/{task}/labelsTr/hcp_{subject}{hemi}.nii.gz',subject=training_subjects, hemi=hemis, allow_missing=True),
        template_json = 'template.json'
    params:
        training_imgs_nosuffix = expand('raw_data/nnUNet_raw_data/{task}/imagesTr/hcp_{subject}{hemi}.nii.gz',subject=training_subjects, hemi=hemis, allow_missing=True),
    output: 
        dataset_json = 'raw_data/nnUNet_raw_data/{task}/dataset.json'
    group: 'preproc'
    script: 'create_json.py' 
    
def get_nnunet_env(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env'].items()])
 
def get_nnunet_env_tmp(wildcards):
     return ' && '.join([f'export {key}={val}' for (key,val) in config['nnunet_env_tmp'].items()])
 
rule plan_preprocess:
    input: 
        dataset_json = 'raw_data/nnUNet_raw_data/{task}/dataset.json'
    params:
        nnunet_env_cmd = get_nnunet_env,
        task_num = lambda wildcards: re.search('Task([0-9]+)\w*',wildcards.task).group(1),
    output: 
        dataset_json = 'preprocessed/{task}/dataset.json'
    group: 'preproc'
    resources:
        threads = 8,
        mem_mb = 16000
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNet_plan_and_preprocess  -t {params.task_num} --verify_dataset_integrity'

def get_checkpoint_opt(wildcards, output):
    if os.path.exists(output.latest_model):
        return '--continue_training'
    else:
        return '' 
      
rule train_fold:
    input:
        dataset_json = 'preprocessed/{task}/dataset.json',
        pretrained_weights = config['pretrained_model']
    params:
        nnunet_env_cmd = get_nnunet_env_tmp,
        rsync_to_tmp = f"rsync -av {config['nnunet_env']['nnUNet_preprocessed']} $TMPDIR",
        #add --continue_training option if a checkpoint exists
        checkpoint_opt = get_checkpoint_opt
    output:
#        model_dir = directory('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}'),
#        final_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_final_checkpoint.model',
        latest_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_latest.model',
        best_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_best.model'
    threads: 16
    resources:
        gpus = 1,
        mem_mb = 64000,
        time = 1440,
    group: 'train'
    shell:
        '{params.nnunet_env_cmd} && '
        '{params.rsync_to_tmp} && '
        'nnUNet_train -pretrained_weights {input.pretrained_weights} {params.checkpoint_opt} {wildcards.arch} {wildcards.trainer} {wildcards.task} {wildcards.fold}'


rule package_trained_model:
    """ Creates tar file for performing inference with workflow_inference -- note, if you do not run training to completion (1000 epochs), then you will need to clear the snakemake metadata before running this rule, else snakemake will not believe that the model has completed. """
    input:
        latest_model = expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',fold=range(5),allow_missing=True),
        latest_model_pkl = expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model.pkl',fold=range(5),allow_missing=True),
        plan = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/plans.pkl'
    params:
        trained_model_dir = config['nnunet_env']['RESULTS_FOLDER'],
        files_to_tar = 'nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1'
    output:
        model_tar = 'trained_model.{arch}.{task}.{trainer}.{checkpoint}.tar'
    shell:
        'tar -cvf {output} -C {params.trained_model_dir} {params.files_to_tar}'


rule predict_test_subj:
    input:
        in_training_folder = expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}',fold=range(5),allow_missing=True),
        latest_model = expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',fold=range(5),allow_missing=True),
        testing_imgs = expand('raw_data/nnUNet_raw_data/{task}/imagesTs/hcp_{subject}{hemi}_0000.nii.gz',subject=testing_subjects, hemi=hemis, allow_missing=True),
    params:
        in_folder = 'raw_data/nnUNet_raw_data/{task}/imagesTs',
        out_folder = 'raw_data/nnUNet_predictions/{arch}/{task}/{trainer}__nnUNetPlansv2.1/{checkpoint}',
        nnunet_env_cmd = get_nnunet_env,
    output:
        testing_imgs = expand('raw_data/nnUNet_predictions/{arch}/{task}/{trainer}__nnUNetPlansv2.1/{checkpoint}/hcp_{subject}{hemi}.nii.gz',subject=testing_subjects, hemi=hemis, allow_missing=True),
    threads: 8 
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 30,
    group: 'inference'
    shell:
        '{params.nnunet_env_cmd} && '
        'nnUNet_predict  -chk {wildcards.checkpoint}  -i {params.in_folder} -o {params.out_folder} -t {wildcards.task}'

   
        

