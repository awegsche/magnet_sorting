#!/bin/bash


MAGNET_SORTING="$(pwd -P)"


# --- setup python ---------------------------------------------------------------------------------
python3 -m venv venv
. venv/bin/activate

pip install matplotlib numpy

git clone git@github.com:pylhc/omc3.git

cd omc3
git checkout enhancement/292/modelcreation

cd ..
pip install -e omc3

# --- setup model output ---------------------------------------------------------------------------
mkdir model

cd model
ln -s ${MAGNET_SORTING}/macros macros
git clone ssh://git@gitlab.cern.ch:7999/acc-models/acc-models-lhc.git
cd acc-models-lhc
git checkout hl16
touch knobs.madx
cd ../..

mkdir model1
cd model1
ln -s ${MAGNET_SORTING}/macros macros
ln -s ${MAGNET_SORTING}/model/acc-models-lhc acc-models-lhc
touch knobs.madx
cd ..

mkdir model10
cd model10
ln -s ${MAGNET_SORTING}/macros macros
ln -s ${MAGNET_SORTING}/model/acc-models-lhc acc-models-lhc
touch knobs.madx
cd ..

mkdir model11
cd model11
ln -s ${MAGNET_SORTING}/macros macros
ln -s ${MAGNET_SORTING}/model/acc-models-lhc acc-models-lhc
touch knobs.madx
cd ..

mkdir model12
cd model12
ln -s ${MAGNET_SORTING}/macros macros
ln -s ${MAGNET_SORTING}/model/acc-models-lhc acc-models-lhc
touch knobs.madx
cd ..

mkdir model13
cd model13
ln -s ${MAGNET_SORTING}/macros macros
ln -s ${MAGNET_SORTING}/model/acc-models-lhc acc-models-lhc
touch knobs.madx
cd ..

echo "pwd = ${MAGNET_SORTING}"
