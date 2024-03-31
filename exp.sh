alg=${1:-'dis'}
game=${2:-'CoinRun'}
ver=${3:-'0.0'}
episodes=${4:-5001}
num_gvfs=${5:-5}

runs=${SLURM_ARRAY_TASK_ID:-0}

if [ "$alg" = "sa_esp" ]; then
        test_args="--game=$game --algorithm=$alg --version=$ver --runs=$runs --train_episodes=$episodes --num_gvfs=$num_gvfs --use_concatanation"
    else
        test_args="--game=$game --algorithm=$alg --version=$ver --runs=$runs --train_episodes=$episodes --num_gvfs=$num_gvfs --use_concatanation --use_action_values --use_off_policy"
    fi

python3 main_procgen.py $test_args