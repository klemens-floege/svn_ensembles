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

