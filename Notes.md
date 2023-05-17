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
* (contacted) Work on a larger dataset (will need to contact Arthur and Adam).
* Contact Arthur and Adam about their progress on modifying the epri21 dataset.
* Remove the --problem argument from the ArgumentParser
* ots_test's generated perturbations don't work for the GIC problem (dimension mismatch).
* Forward function doesn't seem to output binary values for the GIC problem.



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