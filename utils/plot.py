import torch
import matplotlib.pyplot as plt

#TODO:Fix plot function
def plot_modellist(modellist, x_train, y_train, x_test, y_test, save_path):
    """
    Plots predictions from an ensemble of models against true test values and training observations.
    
    Parameters:
    modellist (list): List of PyTorch models.
    x_train (array): Training dataset features.
    y_train (array): Training dataset labels.
    x_test (array): Test dataset features.
    y_test (array): Test dataset labels.
    save_path (str): Path to save the plotted figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))  # Only one plot
    n_particles = len(modellist)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        pred_list = [model.forward(x_test_tensor) for model in modellist]
        y_ensemble = torch.cat(pred_list, dim=0).view(n_particles, -1, 1)
        y_mean = y_ensemble.mean(dim=0).squeeze()
        y_std = y_ensemble.std(dim=0).squeeze()

    x_test_tensor = x_test_tensor.squeeze()

    print('x_test_tesnor: ', x_test_tensor.shape)
    print('x_train: ', x_train.shape)
    print('x_test_tesnor: ', x_test_tensor.shape)
    print('y_mean: ', y_mean.shape)
    print('y_std: ', y_std.shape)

    # Using a distinct color scheme
    ax.plot(x_test_tensor, y_test, color="darkblue", label="True Function")
    ax.scatter(x_train, y_train, s=50, color="darkorange", alpha=0.8, label="Training Data")
    ax.plot(x_test_tensor, y_mean.numpy(), color="green", label="Predicted Mean")
    ax.fill_between(x_test_tensor, (y_mean - y_std).numpy(), (y_mean + y_std).numpy(), color="lightgreen", alpha=0.5, label="Prediction STD")

    for i, member_prediction in enumerate(y_ensemble):
        ax.plot(x_test, member_prediction.numpy().squeeze(), linestyle='--', alpha=0.5, color="grey", label=f"Member {i+1}" if i < 2 else "")

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

'''def plot_modellist(modellist,
                    x_train, y_train,
                    x_test, y_test,
                    save_path
                   ):
    print
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, ax in enumerate(axes):
        
        n_particles = len(modellist)
        
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
 
        #y_ensemble = ensemble.forward(x_test_tensor)

        
        with torch.no_grad():  # Ensure no gradients are computed
            pred_list = []
            for model in modellist:
                pred_list.append(model.forward(x_test_tensor))
            pred = torch.cat(pred_list, dim=0)
            y_ensemble = pred.view(n_particles, -1, 1)  # Adjust dim_problem if necessary

            # Calculate mean and std of predictions
            y_mean = torch.mean(y_ensemble, dim=0).squeeze()  # Mean across particles
            y_std = torch.std(y_ensemble, dim=0).squeeze()  # Std deviation across particles
            
        x_test_tensor = x_test_tensor.squeeze()
        x_test = x_test.squeeze()
        y_test = y_test.squeeze()
        

        ax.plot(x_test, y_test, color="blue", label="sin($2\\pi x$)")
        ax.scatter(x_train, y_train, s=50, alpha=0.5, label="observation")
        
        #ax.plot(x_test_tensor, y_mean, color="red", label="predict mean")
        #ax.plot(x_test_tensor, y_mean, color="red", label="predict mean")
        try:
            ax.plot(x_test_tensor, y_mean, color="red", label="predict mean")
        except Exception as e:
            print("Failed to plot due to an error:", e)


        # Plot ensemble predictions
        for i in range(y_ensemble.shape[0]):
            ax.plot(x_test, y_ensemble[i].numpy().squeeze(), linestyle='--', alpha=0.5, label=f"Member {i+1}")


        ax.fill_between(
            x_test, y_mean - y_std, y_mean + y_std, color="pink", alpha=0.5, label="predict std"
        )
        

    plt.tight_layout()
    
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close()  # Close the plot to free up memory'''