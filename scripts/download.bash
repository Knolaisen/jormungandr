# Create target dir
mkdir -p ./data


download () {
  local url="$1"
  local out="$2"

  # -C - : resume partial download
  # --retry-all-errors : retry on more than just “connection refused”
  # --fail : treat HTTP errors as failures
  curl -L -C - --fail \
    --retry 50 --retry-delay 5 --retry-all-errors \
    --speed-time 30 --speed-limit 10240 \
    -o "$out" "$url"
}
download "http://images.cocodataset.org/zips/train2017.zip" "./data/train2017.zip"
download "http://images.cocodataset.org/zips/val2017.zip"   "./data/val2017.zip"
download "http://images.cocodataset.org/zips/test2017.zip"  "./data/test2017.zip"


unzip -q ./data/train2017.zip -d ./data
unzip -q ./data/val2017.zip -d ./data
unzip -q ./data/test2017.zip -d ./data