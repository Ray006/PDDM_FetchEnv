# PDDM in FetchEnv

## usage ##

We assume that you have installed the related PDDM environment and can run PDDM code successfully.

**Train:**
after git clone:

```bash
cd pddm
```

To run reach task:
```bash
python pddm/scripts/train.py --config ../pddm2/pddm/config/fetchReach_test.txt --output_dir /home/data/Ray_data/pddm_data --use_gpu
```
To run push task:
```bash
python pddm/scripts/train.py --config ../pddm2/pddm/config/fetchPush_test.txt --output_dir /home/data/Ray_data/pddm_data --use_gpu
```
To run slide task:
```bash
python pddm/scripts/train.py --config ../pddm2/pddm/config/fetchSlide_test.txt --output_dir /home/data/Ray_data/pddm_data/testpushnew --use_gpu
```
To run pickandplace task:
```bash
python pddm/scripts/train.py --config ../pddm2/pddm/config/fetchPickAndPlace_test.txt --output_dir /home/data/Ray_data/pddm_data/testpushnew --use_gpu
```
To run FetchEnv with true dynamics(to run different task, you need to modify the env name in the txt document):
```bash
python pddm/scripts/train.py --config ../pddm2/pddm/config/fetch_true_dynamics_h7.txt --output_dir /home/data/Ray_data/pddm_data
```


