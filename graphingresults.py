import numpy as np
import matplotlib.pyplot as plt

# Data preparation
model_sizes = ['Small', 'Medium', 'Large', 'Very Large', 'Extra Large']
learning_rates = [0.001, 0.01, 0.1]
regularizations = ['No regularization', 'L2 (Ridge) - weak', 'L2 (Ridge) - strong', 'L1 (Lasso) - weak', 'L1 (Lasso) - strong']

# Create a 3D numpy array to store the accuracies
accuracies = np.array([
    [[0.9598, 0.9532, 0.9497],  
     [0.9598, 0.9565, 0.9598],  
     [0.9598, 0.9565, 0.9363],  
     [0.9598, 0.9565, 0.9598],  
     [0.9598, 0.9565, 0.9363]], 

    [[0.9598, 0.9565, 0.9564],  
     [0.9598, 0.9565, 0.9564],  
     [0.9598, 0.9565, 0.9599],  
     [0.9598, 0.9565, 0.9564],  
     [0.9598, 0.9565, 0.9599]], 

    [[0.9497, 0.9531, 0.9497],  
     [0.9497, 0.9531, 0.9531],  
     [0.9497, 0.9531, 0.9498],  
     [0.9497, 0.9531, 0.9531],  
     [0.9497, 0.9531, 0.9498]], 

    [[0.9598, 0.9531, 0.9632],  
     [0.9598, 0.9463, 0.9463],  
     [0.9598, 0.9531, 0.9598],  
     [0.9598, 0.9463, 0.9463],  
     [0.9598, 0.9531, 0.9598]], 

    [[0.9565, 0.9598, 0.9632],  
     [0.9599, 0.9598, 0.9496],  
     [0.9599, 0.9599, 0.9463],  
     [0.9599, 0.9598, 0.9496],  
     [0.9599, 0.9599, 0.9463]]  
])

# Define the labels
model_labels = ['Small', 'Medium', 'Large', 'Very Large', 'Extra Large']
reg_labels = ['No regularization', 'L2 (Ridge) - weak', 'L2 (Ridge) - strong', 'L1 (Lasso) - weak', 'L1 (Lasso) - strong']
lr_labels = ['LR 0.001', 'LR 0.01', 'LR 0.1']

# Create the figure and axis
fig, ax = plt.subplots(5, 1, figsize=(10, 20))


# Set the x-axis positions for each regularization technique
x = np.arange(len(reg_labels))

# Set the width of each bar
width = 0.2

# Iterate over the models
for i, model in enumerate(model_labels):
    for j, lr in enumerate(lr_labels):
        ax[i].bar(x + j * width, accuracies[i, :, j], width, label=lr)
    ax[i].set_title(model)
    ax[i].set_xticks(x + width)
    ax[i].set_xticklabels(reg_labels, rotation=45)
    ax[i].set_ylim(0.92, 1)
    ax[i].legend(title='Learning Rate')  # Add a legend
# Layout so plots do not overlap
fig.tight_layout()

fig.savefig('bargraph.png')
plt.show()


