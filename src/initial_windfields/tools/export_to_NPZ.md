# usage static wind

python src/initial_windfields/tools/export_wind_to_npz.py \
  --name simple_static --out winds_export --static true --write-seeds "0,1,2"


  # usage training winds

  python src/initial_windfields/tools/export_wind_to_npz.py \
  --name training_1 --name training_2 --name training_3 \
  --out winds_export --write-seeds "0,1,2"