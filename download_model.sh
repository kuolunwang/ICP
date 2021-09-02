#! /bin/bash

[ ! -d ./model ] && mkdir model

gdown --id 1Rh9P0QyB0ypJHFyFwPNN5mDpr3gd7got

unzip pcd_model.zip -d model

rm pcd_model.zip