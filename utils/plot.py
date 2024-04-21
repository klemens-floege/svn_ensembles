import torch
import matplotlib.pyplot as plt

def plot_modellist(modellist,
                    x_train, y_train,
                    x_test, y_test,
                    save_path
                   ):
    
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
    plt.close()  # Close the plot to free up memory