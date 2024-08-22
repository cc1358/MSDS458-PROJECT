import matplotlib.pyplot as plt
import seaborn as sns

# Data
activations = ['b_spline', 'cubic', 'swish', 'wavelet', 'logistic_regression']
test_accuracy = [55.01, 53.86, 48.43, 39.41, 37.66]
precision = [56.52, 53.25, 47.83, 39.18, 37.15]
recall = [55.01, 53.86, 48.43, 39.41, 37.66]
f1_score = [54.54, 53.13, 47.71, 38.23, 36.99]
runtime = [2731.64, 3703.38, 2374.62, 2541.10, 193.86]

sns.set(style="whitegrid")

# Creating subplots for each metric
fig, axs = plt.subplots(5, 1, figsize=(10, 25))

# Bar chart for Test Accuracy
sns.barplot(x=activations, y=test_accuracy, palette="Blues_d", ax=axs[0])
axs[0].set_title('Test Accuracy')
axs[0].set_ylabel('Percentage (%)')

# Bar chart for Precision
sns.barplot(x=activations, y=precision, palette="Greens_d", ax=axs[1])
axs[1].set_title('Precision')
axs[1].set_ylabel('Percentage (%)')

# Bar chart for Recall
sns.barplot(x=activations, y=recall, palette="Oranges_d", ax=axs[2])
axs[2].set_title('Recall')
axs[2].set_ylabel('Percentage (%)')

# Bar chart for F1 Score
sns.barplot(x=activations, y=f1_score, palette="Reds_d", ax=axs[3])
axs[3].set_title('F1 Score')
axs[3].set_ylabel('Percentage (%)')

# Bar chart for Runtime
sns.barplot(x=activations, y=runtime, palette="Purples_d", ax=axs[4])
axs[4].set_title('Runtime')
axs[4].set_ylabel('Time (seconds)')

plt.tight_layout()
plt.show()
