import json

#load template json
dataset=snakemake.config['dataset_json']


dataset['training'] = [{'image': img, 'label': lbl} for img,lbl in zip(snakemake.params.training_imgs_nosuffix,snakemake.input.training_lbls)]
    
dataset['numTraining'] = len(dataset['training'])

#write modified json
with open(snakemake.output.dataset_json, 'w') as f:
    json.dump(dataset, f, indent=4)
