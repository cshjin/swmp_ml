Things I tried for each figure
0: HGTConv
1: Added another set of ReLU and Linear functions (3 total)
2: Added another 2 set of ReLU and Linear functions (5 total)
3: Same as before
4: Increased hidden_size (hidden layers) from 64 to 128
5: Went back to 64 hidden layers
6: Go to 4 ReLU and Linear functions instead
7: Go back to 5 ReLU and Linear functions and replace the ReLU functions with Sigmoid functions
8: Same as before
9: Added BatchNorm1d between each Sigmoid and Linear function (and rounded the values to 3 decimal places)
10: Reverted back to ReLU functions
11: Same as before
12: Learning rate set to 1e-4 instead of 1e-3
13: Learning rate set to 1e-2

    # Count the number of files that exist in the Figures directory, so
    # we can give a unique name to the two new figures we're creating
    losses_count = len([file_name for file_name in os.listdir('./Figures/Losses/')])
    predictions_count = len([file_name for file_name in os.listdir('./Figures/Predictions/')])

    # Evaluate the model
    model.eval()
    for data in data_loader_test:
        pred = model(data)
        plt.plot(data['y'], "r.", label="true")
        loss = F.mse_loss(pred, data['y'])
        print(loss.item())
        plt.plot(pred.detach().cpu().numpy(), "b.", label="pred")
        plt.legend()
        plt.savefig(f"Figures/Predictions/result_{args['problem']}_{predictions_count}.png")
        break

    ''' plot the loss function '''
    plt.clf()
    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    foo = r'training loss'
    plt.plot(losses)
    plt.ylabel(foo)
    plt.xlabel("epoch")
    plt.title(f"Hete-Graph - {args['problem']}")
    plt.savefig(f"Figures/Losses/losses - {args['problem']}_{losses_count}_final-training-loss={round(t_loss, 3)}_test-loss={round(loss.item(), 3)}.png")


Uninstall commands
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
sudo rm -rf /usr/local/cuda*

pip uninstall torch torchvision torchaudio
pip uninstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric


Reinstall commands
CPU
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install pandas deephyper

CUDA
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html


Reinstall CUDA itself
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


TODO:
- (finished) Add num_mlp_layers support
- (finished) Add the activation functions as a hyperparameters
- (finished) Make a 1 slide summary of Hongwei's presentation
- Try out the other datasets like ots_test, UIUC-150, etc.
- Remove the hard-coded 19 and replace it with the variable for the number of busses


Existing bugs
- dataset.py files have the line "self.name = names[0]." This is to make sure any accesses to self.processed_paths[0] don't crash.
  This does cause the dataset variable in the demo_train files to be named the first power grid in the argument passed into --names,
  but otherwise.
> Solved: bring back MultiGMD

- "pbar = tqdm(range(args['epochs']), desc=args['name'])" has been changed to "pbar = tqdm(range(args['epochs']))"
> Solved.

- Some lines of code for the GIC code in the training loop have been commented out because they cause errors. For example,
  the one about MSE loss.
> Ignore the MSE in the GIC problem.

- hps_mld.py has duplicate code for dataset and data because I'm not sure how to pass data in as an argument for the run() function
> Solved. It's fine to have duplicate code because the run() function can't access variables outside of it.

- When executing the code for the GIC problem, parts of the code dealing with pbar won't execute.
> Solved.

Best results:
p:activation         leakyrelu
p:batch_size               256
p:conv_type                hgt
p:dropout                  0.2
p:hidden_size               16
p:lr                  0.000832
p:num_conv_layers            6
p:num_heads                  1
p:num_mlp_layers             8
p:weight_decay             0.0
objective             0.008532


<!-- DEBUG -->
```
Exception has occurred: ValueError
With n_samples=1, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.
  File "/mnt/c/Users/TurtleCamera/Documents/GitHub/swmp_ml/demo_train.py", line 123, in <module>
    dataset_train, dataset_val = train_test_split(dataset_train,
ValueError: With n_samples=1, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.
```

two grids:
  * 500 samples per grid

double check the multigmd class:
  * `data_list.append(data)` with hetero_data

['b4gic', 'b6gic_nerc', 'ieee_rts_0', 'epri21', 'ots_test', 'uiuc150_95pct_loading', 'uiuc150']

# Best results:
# activation           sigmoid
# batch_size               256
# conv_type                hgt
# dropout                  0.1
# hidden_size              256
# lr                   0.00206
# num_conv_layers            8
# num_heads                  1
# num_mlp_layers             7
# weight_decay          0.0001
# objective            0.008543

# python demo_train.py --problem clf --force --names epri21 --setting gic --activation relu --batch_size 256 --conv_type hgt --dropout 0.1 --hidden_size 256 --lr 1e-3 --num_conv_layers 8 --num_heads 1 --num_mlp_layers 7 --weight_decay 1e-4 --epochs 250


     Single objective (all 3)      Multiple objectives
   |--------------------------|--------------------------|
   |                          |                          |
   |                          |                          |
   |                          |                          |
   |        -test_loss        |                          |
   |         test_acc         |  Tuple of 3 objectives   | GMD (single network)
   |         roc_auc          |                          |
   |                          |                          |
   |                          |                          |
   |                          |                          |
   |--------------------------|--------------------------|
   |                          |                          |
   |                          |                          |
   |                          |                          |
   |        -test_loss        |                          |
   |         test_acc         |  Tuple of 3 objectives   | MultiGMD (multiple networks)
   |         roc_auc          |                          |
   |                          |                          |
   |                          |                          |
   |                          |                          |
   |--------------------------|--------------------------|

  In each case above, evaluate the hyperparameters using DeepHyper, then plug those hyperparameters into demo_train.py to generate figures (12 total).

  New error:
  Traceback (most recent call last):
  File "/mnt/c/Users/TurtleCamera/Documents/GitHub/swmp_ml/hps.py", line 240, in <module>
    results = search.search(max_evals=50)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/search/_search.py", line 131, in search
    self._search(max_evals, timeout)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/search/hps/_cbo.py", line 319, in _search
    new_results = self._evaluator.gather(self._gather_type, size=1)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/evaluator/_evaluator.py", line 325, in gather
    job = task.result()
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/evaluator/_evaluator.py", line 263, in _execute
    job = await self.execute(job)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/evaluator/_thread_pool.py", line 64, in execute
    output = await self.loop.run_in_executor(self.executor, run_function)
  File "/root/miniconda3/lib/python3.10/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/mnt/c/Users/TurtleCamera/Documents/GitHub/swmp_ml/hps.py", line 153, in run
    roc_auc = roc_auc_score(test_y.detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 572, in roc_auc_score
    return _average_binary_score(
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_base.py", line 75, in _average_binary_score
    return binary_metric(y_true, y_score, sample_weight=sample_weight)
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 339, in _binary_roc_auc_score
    raise ValueError(
ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

MultiGMD with multiple objectives:
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  objective_0  objective_1  objective_2  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...    -0.050251     0.398634     0.624928       0            0.027917           29.424206
1    leakyrelu            16         hgt        0.5             64  0.000018  ...    -0.205362     0.457763     0.585713       1           29.513405           86.160908
2          elu            16         han        0.3             16  0.002567  ...    -0.218028     0.694830     0.634313       2           86.200097          374.244642
3    leakyrelu            64         han        0.0            256  0.000750  ...    -0.054160     0.499640     0.562293       3          374.286216          403.615741
4         relu           128         han        0.3             16  0.005596  ...    -0.008671     0.708300     0.558371       4          403.656773          446.970937
5    leakyrelu           128         han        0.0             16  0.000679  ...    -0.008972     0.502551     0.521852       5          447.067812          480.461653
6         tanh           256         hgt        0.3            128  0.009304  ...    -0.008020     0.395034     0.612725       6          480.503976          533.812171
7    leakyrelu           128         han        0.4            256  0.000010  ...    -0.008634     0.488169     0.489424       7          533.855901          569.101483
8         relu            16         hgt        0.4            128  0.000010  ...    -0.204109     0.401869     0.595499       8          569.141772          619.224007
9    leakyrelu            16         hgt        0.3             64  0.004226  ...    -0.200864     0.391453     0.591721       9          619.328738         1048.890131

[10 rows x 16 columns]
Best result for -test_loss:
p:activation              elu
p:batch_size               16
p:conv_type               han
p:dropout                 0.3
p:hidden_size              16
p:lr                 0.002567
p:num_conv_layers           9
p:num_heads                 8
p:num_mlp_layers            2
p:weight_decay        0.00001
objective_0         -0.218028
objective_1           0.69483
objective_2          0.634313
Name: 2, dtype: object 

Best results for test_acc:
p:activation         leakyrelu
p:batch_size                16
p:conv_type                hgt
p:dropout                  0.3
p:hidden_size               64
p:lr                  0.004226
p:num_conv_layers            7
p:num_heads                  1
p:num_mlp_layers             7
p:weight_decay         0.00005
objective_0          -0.200864
objective_1           0.391453
objective_2           0.591721
Name: 9, dtype: object 

Best results for roc_auc score:
p:activation         leakyrelu
p:batch_size               128
p:conv_type                han
p:dropout                  0.4
p:hidden_size              256
p:lr                   0.00001
p:num_conv_layers            6
p:num_heads                  1
p:num_mlp_layers             6
p:weight_decay         0.00001
objective_0          -0.008634
objective_1           0.488169
objective_2           0.489424



GMD (epri21) with multiple objectives:
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  objective_0  objective_1  objective_2  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...    -0.016140     0.660880     0.659964       0            0.012321            8.928146
1          elu            16         han        0.5             64  0.001900  ...    -0.172629     0.539352     0.625444       1            8.972763           36.348898
2         relu            64         hgt        0.0            128  0.000545  ...    -0.016111     0.656250     0.645564       2           36.463741           47.851103
3         relu            16         hgt        0.5             64  0.000588  ...    -0.156950     0.671296     0.664485       3           47.897971           71.075439
4      sigmoid           256         han        0.2             64  0.005091  ...    -0.017125     0.465741     0.567254       4           71.115324           77.066833
5      sigmoid            64         hgt        0.3             16  0.000709  ...    -0.016044     0.625000     0.631313       5           77.108660           88.048328
6    leakyrelu            64         han        0.0             16  0.000843  ...    -0.017126     0.449074     0.555485       6           88.093048           96.304639
7      sigmoid            16         han        0.2             32  0.029213  ...    -0.170651     0.460648     0.561499       7           96.346628          122.985814
8    leakyrelu           128         hgt        0.2             64  0.067470  ...    -0.016037     0.665509     0.677776       8          123.026588          130.954333
9         tanh            16         hgt        0.5             16  0.010320  ...    -0.155449     0.668981     0.671153       9          130.999101          161.097133

[10 rows x 16 columns]
Best result for -test_loss:
p:activation              elu
p:batch_size               16
p:conv_type               han
p:dropout                 0.5
p:hidden_size              64
p:lr                   0.0019
p:num_conv_layers           5
p:num_heads                 8
p:num_mlp_layers            1
p:weight_decay         0.0001
objective_0         -0.172629
objective_1          0.539352
objective_2          0.625444
Name: 1, dtype: object 

Best results for test_acc:
p:activation         leakyrelu
p:batch_size                64
p:conv_type                han
p:dropout                  0.0
p:hidden_size               16
p:lr                  0.000843
p:num_conv_layers            5
p:num_heads                  1
p:num_mlp_layers             1
p:weight_decay         0.00001
objective_0          -0.017126
objective_1           0.449074
objective_2           0.555485
Name: 6, dtype: object 

Best results for roc_auc score:
p:activation         leakyrelu
p:batch_size                64
p:conv_type                han
p:dropout                  0.0
p:hidden_size               16
p:lr                  0.000843
p:num_conv_layers            5
p:num_heads                  1
p:num_mlp_layers             1
p:weight_decay         0.00001
objective_0          -0.017126
objective_1           0.449074
objective_2           0.555485



MultiGMD with single objective (-test_loss):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  objective_0  objective_1  objective_2  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...    -0.016258     0.403921     0.585911       0            0.199494           22.124603
1          elu            16         hgt        0.4            256  0.000012  ...    -0.163772     0.395782     0.539330       1           22.164755           51.319814
2    leakyrelu            64         han        0.0             32  0.000022  ...    -7.266990     0.015295     0.500000       2           51.363699           77.057379
3      sigmoid           128         hgt        0.2             16  0.000021  ...    -0.017119     0.253198     0.561501       3           77.141037          116.403909
4          elu            32         han        0.0             32  0.000116  ...    -0.117377     0.493187     0.519426       4          116.445439          148.086640
5    leakyrelu            16         hgt        0.2             32  0.005998  ...    -0.160728     0.408189     0.581522       5          148.128299          287.828541
6         relu            64         han        0.2             64  0.051142  ...    -0.017345     0.442158     0.504508       6          287.871234          306.220763
7      sigmoid            64         hgt        0.2             64  0.047703  ...    -0.016375     0.121385     0.553301       7          306.336064          344.033757
8          elu            16         han        0.4            128  0.027243  ...    -0.173287     0.019541     0.500000       8          344.076669          374.250315
9          elu            16         han        0.1             64  0.000100  ...    -0.175189     0.480459     0.526215       9          374.294765          428.379612

[10 rows x 16 columns]
Best result for -test_loss:
p:activation         leakyrelu
p:batch_size                64
p:conv_type                han
p:dropout                  0.0
p:hidden_size               32
p:lr                  0.000022
p:num_conv_layers            1
p:num_heads                  1
p:num_mlp_layers             8
p:weight_decay          0.0001
objective_0           -7.26699
objective_1           0.015295
objective_2                0.5
Name: 2, dtype: object 

Best results for test_acc:
p:activation         leakyrelu
p:batch_size                64
p:conv_type                han
p:dropout                  0.0
p:hidden_size               32
p:lr                  0.000022
p:num_conv_layers            1
p:num_heads                  1
p:num_mlp_layers             8
p:weight_decay          0.0001
objective_0           -7.26699
objective_1           0.015295
objective_2                0.5
Name: 2, dtype: object 

Best results for roc_auc score:
p:activation         leakyrelu
p:batch_size                64
p:conv_type                han
p:dropout                  0.0
p:hidden_size               32
p:lr                  0.000022
p:num_conv_layers            1
p:num_heads                  1
p:num_mlp_layers             8
p:weight_decay          0.0001
objective_0           -7.26699
objective_1           0.015295
objective_2                0.5
Name: 2, dtype: object 



MultiGMD with single objective (-test_loss):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000  -0.050251       0            0.020003           62.002378
1          elu           128         han        0.0             16  0.078017  ...                 2         0.00005  -0.008801       1           62.147771          121.830827
2         relu            32         han        0.4             16  0.000016  ...                 7         0.00005  -0.086553       2          121.959377          228.681466
3          elu           256         han        0.2             16  0.000062  ...                 5         0.00010  -0.008655       3          228.773888          301.204030
4    leakyrelu            32         hgt        0.5             16  0.000015  ...                 5         0.00005  -0.082694       4          301.286911          388.614861
5    leakyrelu            64         han        0.5            128  0.000012  ...                 9         0.00001  -0.054507       5          388.771147          478.605309
6    leakyrelu            64         han        0.5             64  0.055032  ...                10         0.00010  -0.054152       6          478.686075          646.074331
7         tanh            64         hgt        0.0             32  0.007593  ...                 3         0.00001  -0.051457       7          646.160563          737.492168
8    leakyrelu            16         han        0.0             16  0.054156  ...                 7         0.00005  -0.216609       8          737.576366         1255.183918
9      sigmoid            32         hgt        0.1             32  0.021800  ...                 4         0.00000  -0.080530       9         1255.289686         1318.356506

[10 rows x 14 columns]
Best results:
p:activation         leakyrelu
p:batch_size                16
p:conv_type                han
p:dropout                  0.0
p:hidden_size               16
p:lr                  0.054156
p:num_conv_layers            3
p:num_heads                  2
p:num_mlp_layers             7
p:weight_decay         0.00005
objective            -0.216609
Name: 8, dtype: object

python demo_train.py --force --names epri21 uiuc150 --setting gic --activation leakyrelu --batch_size 16 --conv_type han --dropout 0.0 --hidden_size 16 --lr 5e-2 --num_conv_layers 3 --num_heads 4 --num_mlp_layers 7 --weight_decay 5e-1 --epochs 250 --weight

Traceback (most recent call last):
  File "/mnt/c/Users/TurtleCamera/Documents/GitHub/swmp_ml/demo_train.py", line 131, in <module>
    roc_auc = roc_auc_score(train_y.detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 572, in roc_auc_score
    return _average_binary_score(
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_base.py", line 75, in _average_binary_score
    return binary_metric(y_true, y_score, sample_weight=sample_weight)
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 339, in _binary_roc_auc_score
    raise ValueError(
ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.





MultiGMD with single objective (test_acc):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000   0.398634       0            0.016402           57.935784
1         relu           128         hgt        0.4             64  0.000474  ...                10         0.00001   0.400895       1           58.097811          139.129289
2      sigmoid            16         hgt        0.0             16  0.093515  ...                 6         0.00010   0.964818       2          139.185980          645.363853
3    leakyrelu            64         hgt        0.1             16  0.000020  ...                 9         0.00005   0.418053       3          645.439116          709.537873
4    leakyrelu           128         hgt        0.4             16  0.000830  ...                 6         0.00005   0.400565       4          709.585986          757.583551
5         tanh           128         hgt        0.2             64  0.044324  ...                 5         0.00005   0.404978       5          757.705285          822.090981
6    leakyrelu            16         hgt        0.3             32  0.000261  ...                 9         0.00005   0.386676       6          822.138163         1014.778477
7          elu           256         han        0.5            256  0.000197  ...                 9         0.00000   0.514279       7         1014.827004         1068.934259
8      sigmoid           256         hgt        0.5             16  0.002798  ...                10         0.00000   0.414597       8         1068.985958         1144.535792
9         relu            32         hgt        0.2            256  0.000067  ...                 6         0.00000   0.406729       9         1144.674752         1221.778575

[10 rows x 14 columns]
Best results:
p:activation         leakyrelu
p:batch_size                16
p:conv_type                hgt
p:dropout                  0.3
p:hidden_size               32
p:lr                  0.000261
p:num_conv_layers            9
p:num_heads                  8
p:num_mlp_layers             9
p:weight_decay         0.00005
objective             0.386676
Name: 6, dtype: object

python demo_train.py --force --names epri21 uiuc150 --setting gic --activation leakyrelu --batch_size 16 --conv_type hgt --dropout 0.3 --hidden_size 32 --lr 3e-4 --num_conv_layers 9 --num_heads 8 --num_mlp_layers 9 --weight_decay 5e-1 --epochs 250 --weight

Traceback (most recent call last):
  File "/mnt/c/Users/TurtleCamera/Documents/GitHub/swmp_ml/demo_train.py", line 131, in <module>
    roc_auc = roc_auc_score(train_y.detach().cpu().numpy(), out.argmax(1).detach().cpu().numpy())
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 572, in roc_auc_score
    return _average_binary_score(
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_base.py", line 75, in _average_binary_score
    return binary_metric(y_true, y_score, sample_weight=sample_weight)
  File "/root/miniconda3/lib/python3.10/site-packages/sklearn/metrics/_ranking.py", line 339, in _binary_roc_auc_score
    raise ValueError(
ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.





MultiGMD with single objective (roc_auc):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000   0.624928       0            0.016981           55.948541
1    leakyrelu            64         han        0.0             64  0.000144  ...                 8         0.00001   0.525727       1           56.110152          122.853852
2    leakyrelu           128         hgt        0.2             16  0.007937  ...                10         0.00001   0.624953       2          122.956356          240.526681
3          elu            16         han        0.4             32  0.015443  ...                 5         0.00010   0.535258       3          240.603022          581.252377
4          elu            16         hgt        0.2             64  0.000036  ...                 2         0.00010   0.611111       4          581.336823          715.874599
5          elu           128         hgt        0.2             64  0.042347  ...                 3         0.00010   0.608737       5          716.094794          812.298707
6          elu            16         hgt        0.5             16  0.000013  ...                 3         0.00000   0.605907       6          812.386194          936.896884
7         tanh            32         hgt        0.5             16  0.008692  ...                10         0.00010   0.619749       7          936.978073         1144.501723
8          elu           256         han        0.3            256  0.046692  ...                 5         0.00000   0.561718       8         1144.594028         1229.898732
9         relu            64         hgt        0.2            128  0.003839  ...                 1         0.00005   0.605512       9         1230.106672         1323.903686

[10 rows x 14 columns]
Best results:
p:activation         leakyrelu
p:batch_size                64
p:conv_type                han
p:dropout                  0.0
p:hidden_size               64
p:lr                  0.000144
p:num_conv_layers            7
p:num_heads                  4
p:num_mlp_layers             8
p:weight_decay         0.00001
objective             0.525727
Name: 1, dtype: object

python demo_train.py --force --names epri21 uiuc150 --setting gic --activation leakyrelu --batch_size 64 --conv_type han --dropout 0.0 --hidden_size 64 --lr 1e-4 --num_conv_layers 7 --num_heads 4 --num_mlp_layers 8 --weight_decay 1e-5 --epochs 250 --weight

Weighted loss: 2.4240154027938843
Accuracy: 0.2722710163111669
ROC_AUC score: 0.6152124773960217






GMD (epri21) with single objective (-test_loss):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000  -0.016140       0            0.021120           20.161131
1         relu            32         hgt        0.5             64  0.021338  ...                 8         0.00005  -0.106461       1           20.265389           57.454382
2          elu            64         han        0.0            128  0.000014  ...                 8         0.00005  -8.436924       2           57.531701           81.519746
3         relu            64         han        0.4             16  0.001197  ...                10         0.00001  -0.017302       3           81.601357          108.159070
4         tanh           256         hgt        0.1            256  0.000012  ...                 7         0.00001  -0.017393       4          108.241783          126.356143
5         tanh            64         han        0.0             16  0.000143  ...                 2         0.00000  -0.017135       5          126.438666          147.986255
6      sigmoid            16         han        0.0             64  0.000016  ...                 3         0.00010  -0.171135       6          148.154914          180.931981
7         relu            64         han        0.1             32  0.000013  ...                 2         0.00000  -0.017364       7          181.008877          202.654645
8         relu            32         han        0.1            256  0.011459  ...                 1         0.00000  -0.108954       8          202.736121          225.200268
9         relu           128         han        0.2             64  0.006034  ...                 6         0.00000  -0.017132       9          225.269996          242.568644

[10 rows x 14 columns]
Best results:
p:activation              elu
p:batch_size               64
p:conv_type               han
p:dropout                 0.0
p:hidden_size             128
p:lr                 0.000014
p:num_conv_layers           7
p:num_heads                 8
p:num_mlp_layers            8
p:weight_decay        0.00005
objective           -8.436924
Name: 2, dtype: object

python demo_train.py --force --names epri21 --setting gic --activation elu --batch_size 64 --conv_type han --dropout 0.0 --hidden_size 128 --lr 5e-1 --num_conv_layers 7 --num_heads 8 --num_mlp_layers 8 --weight_decay 5e-5 --epochs 250 --weight

Weighted loss: 1.3883929252624512
Accuracy: 0.73046875
ROC_AUC score: 0.37250996015936255





GMD (epri21) with single objective (test_acc):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000   0.660880       0            0.026382           10.134376
1         tanh            32         hgt        0.1             32  0.000455  ...                 1         0.00001   0.663194       1           10.183912           22.032282
2    leakyrelu            64         hgt        0.4             64  0.000824  ...                 3         0.00005   0.657407       2           22.075586           33.400249
3    leakyrelu           256         hgt        0.3             64  0.000019  ...                 4         0.00010   0.475694       3           33.446848           45.020631
4         tanh            64         hgt        0.4             64  0.000028  ...                10         0.00005   0.469907       4           45.105651           63.380012
5      sigmoid           128         han        0.2             32  0.003631  ...                 7         0.00001   0.592593       5           63.456269           82.050347
6         tanh           256         hgt        0.3            256  0.001710  ...                 5         0.00001   0.648843       6           82.140580           96.628134
7    leakyrelu            16         hgt        0.1             64  0.005294  ...                 5         0.00010   0.666667       7           96.706455          160.541138
8         tanh           128         han        0.3            256  0.032511  ...                 4         0.00001   0.453704       8          160.614290          179.483530
9         relu            16         hgt        0.2            128  0.000357  ...                 6         0.00000   0.634259       9          179.571147          211.100739

[10 rows x 14 columns]
Best results:
p:activation             tanh
p:batch_size              128
p:conv_type               han
p:dropout                 0.3
p:hidden_size             256
p:lr                 0.032511
p:num_conv_layers           2
p:num_heads                 8
p:num_mlp_layers            4
p:weight_decay        0.00001
objective            0.453704

python demo_train.py --force --names epri21 --setting gic --activation tanh --batch_size 128 --conv_type han --dropout 0.3 --hidden_size 256 --lr 3e-2 --num_conv_layers 2 --num_heads 8 --num_mlp_layers 4 --weight_decay 1e-5 --epochs 250 --weight

Weighted loss: 0.6943098306655884
Accuracy: 0.556640625
ROC_AUC score: 0.5192792792792793





GMD (epri21) with single objective (roc_auc):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000   0.659964       0            0.012786           10.853943
1          elu           256         hgt        0.0            128  0.000086  ...                 1         0.00001   0.649567       1           10.907465           18.450899
2      sigmoid           128         han        0.0            128  0.011610  ...                 1         0.00005   0.557381       2           18.498725           26.730350
3         relu           128         han        0.2            256  0.000018  ...                 7         0.00005   0.486370       3           26.779604           37.246799
4    leakyrelu            32         hgt        0.1            256  0.000022  ...                 4         0.00010   0.698145       4           37.297894           57.506499
5    leakyrelu           256         hgt        0.0             64  0.000347  ...                 4         0.00001   0.661799       5           57.559544           67.004689
6    leakyrelu            16         han        0.3             64  0.000024  ...                10         0.00010   0.558201       6           67.065918           94.578472
7      sigmoid            32         hgt        0.3            128  0.000032  ...                 7         0.00010   0.638037       7           94.634661          112.212439
8          elu            16         hgt        0.4            256  0.014255  ...                 3         0.00000   0.672065       8          112.263538          134.021358
9    leakyrelu            64         han        0.2             16  0.002194  ...                 4         0.00010   0.607894       9          134.067380          145.741367

[10 rows x 14 columns]
Best results:
p:activation             relu
p:batch_size              128
p:conv_type               han
p:dropout                 0.2
p:hidden_size             256
p:lr                 0.000018
p:num_conv_layers           4
p:num_heads                 4
p:num_mlp_layers            7
p:weight_decay        0.00005
objective             0.48637

python demo_train.py --force --names epri21 --setting gic --activation relu --batch_size 128 --conv_type han --dropout 0.2 --hidden_size 256 --lr 2e-5 --num_conv_layers 4 --num_heads 4 --num_mlp_layers 7 --weight_decay 5e-5 --epochs 250 --weight

Weighted loss: 0.6716804504394531
Accuracy: 0.5498046875
ROC_AUC score: 0.5937737737737738





GMD (uiuc150) with single objective (-test_loss):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000  -0.016258       0            0.013112           32.279124
1    leakyrelu           128         hgt        0.5            256  0.000821  ...                10         0.00000  -0.017127       1           32.334691           79.230566
2      sigmoid            32         hgt        0.0            128  0.010021  ...                 4         0.00010  -0.100957       2           79.280674          130.015333
3         relu           128         han        0.1            256  0.000237  ...                 2         0.00001  -0.017347       3          130.126022          153.747140
4         tanh            16         han        0.5            256  0.000176  ...                10         0.00005  -0.173360       4          153.797899          222.425088
5    leakyrelu           256         hgt        0.1             64  0.000046  ...                 8         0.00001  -0.016426       5          222.475383          263.920538
6    leakyrelu            64         hgt        0.3             16  0.000151  ...                 3         0.00001  -0.016223       6          263.967187          296.484179
7      sigmoid            16         han        0.1             64  0.007580  ...                 1         0.00001  -0.173358       7          296.625110          364.718426
8      sigmoid            64         hgt        0.3             32  0.019084  ...                 4         0.00005  -0.016224       8          364.765601          414.507416
9      sigmoid            64         hgt        0.0             32  0.074767  ...                 1         0.00001  -0.016244       9          414.561484          456.018808

[10 rows x 14 columns]
Best results:
p:activation             tanh
p:batch_size               16
p:conv_type               han
p:dropout                 0.5
p:hidden_size             256
p:lr                 0.000176
p:num_conv_layers           4
p:num_heads                 1
p:num_mlp_layers           10
p:weight_decay        0.00005
objective            -0.17336
Name: 4, dtype: object

python demo_train.py --force --names uiuc150 --setting gic --activation tanh --batch_size 16 --conv_type han --dropout 0.5 --hidden_size 256 --lr 2e-4 --num_conv_layers 4 --num_heads 1 --num_mlp_layers 10 --weight_decay 5e-5 --epochs 250 --weight

Weighted loss: 5.039179444313049
Accuracy: 0.21585557299843014
ROC_AUC score: 0.6054502369668247





GMD (uiuc150) with single objective (test_acc):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000   0.403921       0            0.014806           29.435664
1      sigmoid            16         han        0.4            128  0.005375  ...                 7         0.00001   0.744417       1           29.485760          127.648195
2          elu            64         hgt        0.3             64  0.029538  ...                 4         0.00001   0.363598       2          127.695122          173.850858
3    leakyrelu            32         han        0.3             64  0.000099  ...                10         0.00001   0.398359       3          173.964107          217.682334
4    leakyrelu           128         hgt        0.3            128  0.002624  ...                 2         0.00000   0.401696       4          217.732155          246.604750
5          elu            16         han        0.4            256  0.014409  ...                 4         0.00005   0.297767       5          246.650619          328.824350
6    leakyrelu           256         hgt        0.5            256  0.000533  ...                 5         0.00005   0.331185       6          328.866353          359.977724
7      sigmoid            64         han        0.2             64  0.023212  ...                10         0.00001   0.461207       7          360.084707          402.670711
8         relu           128         hgt        0.2            128  0.012955  ...                 5         0.00010   0.404060       8          402.720413          448.579516
9    leakyrelu            64         hgt        0.0            256  0.003962  ...                 5         0.00000   0.402113       9          448.626113          479.271146

[10 rows x 14 columns]
Best results:
p:activation              elu
p:batch_size               16
p:conv_type               han
p:dropout                 0.4
p:hidden_size             256
p:lr                 0.014409
p:num_conv_layers           2
p:num_heads                 4
p:num_mlp_layers            4
p:weight_decay        0.00005
objective            0.297767
Name: 5, dtype: object

python demo_train.py --force --names uiuc150 --setting gic --activation elu --batch_size 16 --conv_type han --dropout 0.4 --hidden_size 256 --lr 1e-2 --num_conv_layers 2 --num_heads 4 --num_mlp_layers 4 --weight_decay 5e-5 --epochs 250 --weight

Weighted loss: 5.039179444313049
Accuracy: 0.21585557299843014
ROC_AUC score: 0.6054502369668247





GMD (uiuc150) with single objective (roc_auc):
  p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  p:num_mlp_layers  p:weight_decay  objective  job_id  m:timestamp_submit  m:timestamp_gather
0         relu            64         hgt        0.5            128  0.001000  ...                 1         0.00000   0.585911       0            0.014987           33.590289
1         tanh           128         hgt        0.1             64  0.000057  ...                 8         0.00001   0.567873       1           33.656851           85.046138
2      sigmoid           128         han        0.4             32  0.006912  ...                 3         0.00010   0.524442       2           85.098099          112.576741
3      sigmoid            64         han        0.4            128  0.000447  ...                 9         0.00000   0.562509       3          112.710515          151.882416
4         tanh           128         han        0.2            256  0.000121  ...                 8         0.00000   0.487218       4          151.931359          186.648051
5    leakyrelu            32         hgt        0.5             64  0.089794  ...                 1         0.00010   0.587204       5          186.699530          232.022749
6    leakyrelu            16         hgt        0.0             64  0.064710  ...                 9         0.00010   0.581218       6          232.068126          359.169720
7    leakyrelu            64         han        0.5            128  0.000116  ...                 4         0.00001   0.514902       7          359.271012          390.481524
8      sigmoid            32         han        0.4            128  0.000110  ...                10         0.00000   0.505466       8          390.529562          430.465036
9          elu            32         han        0.3            256  0.006757  ...                 4         0.00005   0.571417       9          430.510303          461.356824

[10 rows x 14 columns]
Best results:
p:activation             tanh
p:batch_size              128
p:conv_type               han
p:dropout                 0.2
p:hidden_size             256
p:lr                 0.000121
p:num_conv_layers          10
p:num_heads                 2
p:num_mlp_layers            8
p:weight_decay            0.0
objective            0.487218

python demo_train.py --force --names uiuc150 --setting gic --activation tanh --batch_size 128 --conv_type han --dropout 0.2 --hidden_size 256 --lr 1e-4 --num_conv_layers 10 --num_heads 2 --num_mlp_layers 8 --weight_decay 0.0 --epochs 250 --weight

Processing...
Null in result file
Null in result file
Null in result file
Done!
uiuc150:   0%|                                                                                                                          | 0/250 [00:00<?, ?it/s]
Killed





Error for bounds:

Traceback (most recent call last):
  File "/mnt/c/Users/TurtleCamera/Documents/GitHub/swmp_ml/hps.py", line 247, in <module>
    results = search.search(max_evals=50)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/search/_search.py", line 131, in search
    self._search(max_evals, timeout)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/search/hps/_cbo.py", line 419, in _search
    new_X = self._opt.ask(
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/skopt/optimizer/optimizer.py", line 521, in ask
    x = self._ask()
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/skopt/optimizer/optimizer.py", line 834, in _ask
    [self.space.distance(next_x, xi) for xi in self.Xi]
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/skopt/optimizer/optimizer.py", line 834, in <listcomp>
    [self.space.distance(next_x, xi) for xi in self.Xi]
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/skopt/space/space.py", line 1472, in distance
    distance += dim.distance(a, b)
  File "/root/miniconda3/lib/python3.10/site-packages/deephyper/skopt/space/space.py", line 903, in distance
    raise RuntimeError(
RuntimeError: Can only compute distance for values within the space, not 1 and 64.

TODO:
* > run hps on epri21 for gic problem
* > (has issue) hps with multiple grids
* > Try out normalization in DeepHyper
* > generate perturbations of ots_test for gic problem
* > run demo_train.py with multple grids (epri21 and ots_test)
* > Discuss with Hongwei his progress on the constraints. Hongwei is currently thinking about the contraints that Arthur gave him and formulating a way to incorporate them into the model.
* > Hongwei needs a routine from Russell to evaluate the quality of the blocker placement output from the neural network. Hongwei will email Russell directly about this.
* > Make sure that there are actually blockers being placed and if the output of the model is trying to place blockers. Note that epri21 seems to be missing some data about the current. Follow up with Arthur and Adam about their updates to the epri21 power grid dataset.
* > Take a look at the cross entropy function to see how it handles regression and classification outputs. Specifically, the output of the model (the "out" variable) doesn't seem to have binary values.
* > Forward function doesn't seem to output binary values for the GIC problem.image.png
* > Remove the --problem argument from the ArgumentParser
* > ots_test's generated perturbations don't work for the GIC problem (dimension mismatch).
* > Issue with dimension mismatch in weight tensor
* > Add the ability to evaluate the run function on multiple objectives
* > Come up with a plan to run experiments after fixing the two issues above

* (contacted) Work on a larger dataset (will need to contact Arthur and Adam).
* Contact Arthur and Adam about their progress on modifying the epri21 dataset.
* Test the code that computes roc_auc to see if it fixes the score at 0 for cases where there are only 0's in the output.
* Implement a spider plot to compare multiple objective functions.
* Check the DeepHyper documentation for cross-validation support.



   p:activation  p:batch_size p:conv_type  p:dropout  p:hidden_size      p:lr  ...  objective_0  objective_1  objective_2  job_id  m:timestamp_submit  m:timestamp_gather
0          relu            64         hgt        0.5            128  0.001000  ...    -0.050251     0.402730     0.629141       0            0.027108           58.931639
1       sigmoid            16         han        0.5             32  0.011195  ...    -0.215466     0.498862     0.537635       1           59.005633          376.724852
2          tanh            16         han        0.1            128  0.057682  ...    -0.216609     0.029010     0.500000       2          376.861066         1118.322644
3           elu            16         hgt        0.3              1  0.000832  ...    -0.202964     0.389091     0.593325       3         1118.396046         1445.604377
4           elu           256         hgt        0.0             32  0.024708  ...    -0.042133     0.394989     0.618016       4         1445.663296         1522.365547
5          relu            32         hgt        0.1              1  0.000250  ...    -0.079715     0.397042     0.641159       5         1522.426760         1594.429025
6           elu            64         hgt        0.0             16  0.006004  ...    -0.052846     0.374226     0.588405       6         1594.488064         1687.905017
7          relu           128         han        0.2            128  0.003087  ...    -0.008621     0.501306     0.542543       7         1687.961372         1726.003014
8     leakyrelu             1         hgt        0.5              1  0.026499  ...   -51.916141     0.740741     0.500000       8         1726.061728         5267.849705
9          tanh           128         hgt        0.3              1  0.000403  ...    -0.008201     0.390249     0.615581       9         5268.184360         5355.004903
10         relu             1         han        0.4              1  0.000308  ...   -50.747453     1.000000     0.000000      10         5355.442012         6060.049948
11    leakyrelu             1         han        0.2              1  0.000019  ...   -51.255060     1.000000     0.000000      11         6060.513236         7124.931322
12          elu             1         han        0.0              1  0.002299  ...   -50.529465     1.000000     0.000000      12         7125.541873         8799.923832
13         relu             1         hgt        0.2              1  0.009021  ...   -46.411769     0.925926     0.500000      13         8800.388681        10167.381911
14    leakyrelu             1         han        0.2              1  0.001457  ...   -50.533540     0.851852     0.500000      14        10168.689726        11595.251328
15         relu             1         han        0.0              1  0.000133  ...   -50.402499     1.000000     0.000000      15        11595.731140        13167.328510
16         relu             1         han        0.2              1  0.000061  ...   -50.595054     0.888889     0.500000      16        13168.423155        14673.559703
17    leakyrelu             1         hgt        0.2              1  0.002342  ...   -46.447021     0.925926     0.500000      17        14674.037545        16534.586619
18          elu             1         han        0.2              1  0.000531  ...   -50.519887     0.925926     0.500000      18        16535.113660        18242.276937
19    leakyrelu             1         han        0.3              1  0.000610  ...   -50.662222     0.959677     0.500000      19        18242.781301        19984.222241
20    leakyrelu             1         han        0.2              1  0.001378  ...   -50.538326     0.740741     0.500000      20        19984.864059        21572.344732
21         relu             1         han        0.4              1  0.000012  ...   -51.956192     0.740741     0.500000      21        21573.296349        22487.557123
22          elu             1         han        0.4              1  0.004227  ...   -51.214690     1.000000     0.000000      22        22488.005852        23808.447886
23          elu             1         han        0.3              1  0.000084  ...   -50.647155     0.851852     0.500000      23        23808.988902        25021.622078
24    leakyrelu             1         han        0.1              1  0.000058  ...   -50.500246     0.851852     0.500000      24        25022.019140        26384.144659
25    leakyrelu             1         han        0.0              1  0.001750  ...   -50.500154     0.740741     0.500000      25        26384.524998        27894.111527
26    leakyrelu             1         han        0.4              1  0.002966  ...   -50.696668     0.851852     0.500000      26        27894.499308        29376.211648
27    leakyrelu             1         han        0.2              1  0.000070  ...   -50.560021     0.959677     0.500000      27        29376.623184        30570.468673
28         relu             1         han        0.4              1  0.000039  ...   -50.767966     0.967742     0.500000      28        30570.855214        31568.584764
29          elu             1         han        0.0              1  0.006138  ...   -50.463410     0.851852     0.500000      29        31568.971574        32507.204241
30          elu             1         han        0.5              1  0.002344  ...   -50.793513     0.962963     0.500000      30        32507.559464        33859.982004
31    leakyrelu             1         han        0.2              1  0.000103  ...   -50.554114     1.000000     0.000000      31        33860.408756        35106.252073
32      sigmoid             1         han        0.5              1  0.000728  ...   -50.767567     0.963710     0.500000      32        35106.695311        36578.548183
33    leakyrelu             1         han        0.3              1  0.003067  ...   -50.861674     0.925926     0.500000      33        36578.848047        37844.239756
34         relu             1         han        0.0              1  0.000453  ...   -50.396310     1.000000     0.000000      34        37844.696692        39495.813907
35         relu             1         han        0.1              1  0.000194  ...   -50.481588     0.888889     0.500000      35        39496.422029        41021.200707
36          elu             1         han        0.2              1  0.000273  ...   -50.562331     1.000000     0.000000      36        41021.664629        42120.768217
37    leakyrelu             1         han        0.4              1  0.003708  ...   -50.824198     0.963710     0.500000      37        42121.259719        43183.838295
38          elu             1         han        0.2              1  0.007984  ...   -50.655427     0.851852     0.500000      38        43184.125332        44064.411018
39          elu             1         han        0.4              1  0.000018  ...   -51.190616     0.959677     0.500000      39        44064.704931        45041.042624
40         relu             1         han        0.1              1  0.039568  ...   -51.239077     0.963710     0.500000      40        45041.337323        45670.999961
41         tanh             1         han        0.2              1  0.005312  ...   -50.682371     0.888889     0.500000      41        45671.288549        46623.177025
42    leakyrelu             1         han        0.2              1  0.003764  ...   -50.417070     0.925926     0.500000      42        46623.471417        47665.727061
43          elu             1         han        0.0              1  0.008170  ...   -50.306427     0.963710     0.500000      43        47666.015620        48547.086493
44    leakyrelu             1         han        0.1              1  0.004567  ...   -50.381295     0.888889     0.500000      44        48547.385266        49497.135866
45         relu             1         han        0.2              1  0.000105  ...   -50.575100     0.888889     0.500000      45        49497.430084        50446.871501
46    leakyrelu             1         han        0.1              1  0.000025  ...   -50.765831     1.000000     0.000000      46        50447.224317        51138.686543
47          elu             1         han        0.4              1  0.006681  ...   -50.732239     0.962963     0.500000      47        51138.980629        52033.946784
48         relu             1         han        0.5              1  0.005354  ...   -51.619579     0.888889     0.500000      48        52034.298805        52979.427126
49      sigmoid             1         han        0.0              1  0.003110  ...   -50.291989     0.814815     0.500000      49        52979.717743        54045.582379