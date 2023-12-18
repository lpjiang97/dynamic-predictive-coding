import argparse
import os
import slrm

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data-dir', default='data/64x64_SIGNS', help="Directory containing the dataset")


def launch_training_job(model_dir, data_dir, job_name, params, slrm_kwargs, run=True):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(model_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    if run:
        # Launch training with this config
        cmd = "source /mmfs1/home/prestonj/.zshrc \n" + "home \n" + \
              "cd projects/DynPredCode \n" + \
              "apptainer exec --home $PWD --bind /mmfs1/home/ --bind /mmfs1/gscratch/ --bind ~/.Xauthority --nv /gscratch/rao/prestonj/projects/DynPredCode/predcode.sif " 
        cmd += f"python train_ista.py --model_dir={model_dir} --data_dir {data_dir}"
        print(cmd)
        slrm.launch(cmd, slrm_kwargs)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    #seeds = [1, 10, 250, 500, 567, 678, 704, 899]
    seeds = [678, 704, 899]

    for s_d in seeds:
        # Modify the relevant parameter in params
        params.seed = s_d

        # Launch job (name has to be unique)
        job_name = "seed_{}".format(s_d)
        slrm_kwargs = {
            "job-name": job_name,
            "partition": "ckpt",
            "account": "cse",
            "nodes": "1",
            "cpus-per-task": "4",
            "mem": "30G",
            "time": "12:00:00",
            "gres": "gpu:1",
            "mail-type": "END,FAIL",
            "mail-user": "prestonj@uw.edu"
        }
        launch_training_job(args.model_dir, args.data_dir, job_name, params, slrm_kwargs)

    #mixture_dims = [2, 3, 4, 5, 6, 7, 8]

    #for mix_dim in mixture_dims:
    #    # Modify the relevant parameter in params
    #    params.mix_dim = mix_dim

    #    # Launch job (name has to be unique)
    #    job_name = f"mix_dim_{mix_dim}"
    #    slrm_kwargs = {
    #        "job-name": job_name,
    #        "partition": "ckpt",
    #        "account": "cse",
    #        "nodes": "1",
    #        "cpus-per-task": "4",
    #        "mem": "30G",
    #        "time": "20:00:00",
    #        "gres": "gpu:1",
    #        "mail-type": "END,FAIL",
    #        "mail-user": "prestonj@uw.edu"
    #    }
    #    launch_training_job(args.model_dir, args.data_dir, job_name, params, slrm_kwargs)
