import pandas as pd
import matplotlib.pyplot as plt

# Dictionary with legend names as keys and file paths as values
files_dict = {
    'Evaluative': 'results/evaluative.csv',
    'Binary Preference': 'results/preference.csv',
    'Marginal Binary Preference': 'results/preference_margin.csv',
    'Random': 'results/random.csv',
}

# Customize the plot fonts
plt.rc('font', size=14)         # controls default text sizes
plt.rc('axes', titlesize=18)    # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)   # fontsize of the tick labels
plt.rc('ytick', labelsize=14)   # fontsize of the tick labels
plt.rc('legend', fontsize=16)   # legend fontsize
# Create a plot
plt.figure(figsize=(10, 6))

# Loop through the dictionary, read each CSV file, and plot the data
for legend_name, file_path in files_dict.items():
    df = pd.read_csv(file_path)
    df['value'] = df['0']
    df['step'] = df.index * 100  # Multiply step values by 100
    plt.plot(df['step'], df['value'], label=legend_name)


# Adding title and labels
#plt.title('Training Process')
plt.xlabel('Step', fontsize=16)
plt.ylabel('Total Reward', fontsize=16)

# Adding a legend
plt.legend()

# Save the plot to a file
plt.savefig('training_plot.png')

# Show the plot
plt.show()
