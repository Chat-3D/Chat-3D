export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1
MASTER_NODE='localhost'

stage=1  # change for different stage (1, 2, 3)
epoch=3
pretrained_path=""
evaluate=False
lr=5e-3

while [ "$#" -gt 0 ]; do
    case "$1" in
        --stage)
            stage="$2"
            shift 2
            ;;
        --pretrained_path)
            pretrained_path="$2"
            shift 2
            ;;
        --evaluate)
            evaluate=True
            shift
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ $stage -eq 3 ]; then
  max_txt_len=512
else
  max_txt_len=32
fi

OUTPUT_DIR=outputs/"$(date +"%Y-%m-%d-%T" | tr -d ':')"_sta"$stage"_ep"$epoch"

torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
          --rdzv_endpoint=${MASTER_NODE}:10068 \
          --rdzv_backend=c10d \
          tasks/train.py \
          $(dirname $0)/config.py \
          output_dir ${OUTPUT_DIR} \
          model.stage "$stage" \
          scheduler.epochs "$epoch" \
          optimizer.lr "$lr" \
          model.max_txt_len "$max_txt_len" \
          pretrained_path "$pretrained_path" \
          evaluate "$evaluate"
