set -ue

# Check that runpod volume is mounted and has at least 1GB free
echo "Checking runpod volume..."
if [ ! -d /runpod-volume ]; then
    echo "ERROR: Runpod volume not mounted!"
    exit 1
fi

## Test runpod-volume has at least 7GB free
free_space=$(df -k --output=avail /runpod-volume | tail -n1)
if [ $free_space -lt 7000000 ]; then
    echo "ERROR: Runpod volume has less than 7GB free!"
    exit 1
fi

# Echo free space
echo "Runpod volume OK with $((free_space / 1000000))GB free"

echo "Setting up runpod mounts with unlink..."
rm -rf /data
mkdir -p /runpod-volume/data/data
ln -sf /runpod-volume/data/data /data

mkdir -p /runpod-volume/data/cache
rm -rf /root/.cache
ln -sf /runpod-volume/data/cache /root/.cache

# basicsr for super-resolution compatability fix for torchvision
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/g' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py
