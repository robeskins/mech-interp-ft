GPU_ID=1
THRESHOLD=19000

while true; do
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")

    if [[ -z "$MEM_TOTAL" || -z "$MEM_USED" ]]; then
        echo "Could not retrieve GPU memory. Retrying..."
        sleep 5
        continue
    fi

    MEM_FREE=$((MEM_TOTAL - MEM_USED))
    echo "GPU $GPU_ID has $MEM_FREE MiB free"

    if [ "$MEM_FREE" -ge "$THRESHOLD" ]; then
        echo "Enough memory available. Starting job..."
        break
    else
        echo "Waiting... not enough free memory."
        sleep 10
    fi
done

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py
CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_metrics.py