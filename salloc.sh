#!/bin/bash
salloc --time 4:00:00 --nodes 1 --ntasks-per-node 2 --gpus 2 --mem 512g --qos dw87 --partition dw
