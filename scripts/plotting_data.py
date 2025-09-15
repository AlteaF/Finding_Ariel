import matplotlib.pyplot as plt 

species = ["Fish", "Starfish", "Crab", "Black goby", "Wrasse", "Two-spotted goby", "Cod" , "Painted goby", "Sand eel", "Whiting"]

train_numbers = [9615, 18543, 5213, 3743, 965, 644, 592, 547, 285, 212]
test_numbers = [2619, 4640, 2630, 1009, 247, 227, 170, 132, 39, 49]

fig, axes = plt.subplots(1,2, figsize= (15,10))

axes[0].bar(species, train_numbers, color="red")
axes[0].set_title("Distribution of fish species in train data")
axes[0].set_ylabel("Number of fish")
axes[0].set_xlabel("Fish Species")
axes[0].set_xticklabels(species, rotation=45)


axes[1].bar(species, test_numbers, color="orange")
axes[1].set_title("Distribution of fish species in test data")
axes[1].set_ylabel("Number of fish")
axes[1].set_xlabel("Fish Species")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()