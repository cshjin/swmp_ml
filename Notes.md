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