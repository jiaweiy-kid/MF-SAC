# MF-SAC
## Introduction

This repository contains the code for the paper **“[Toward Electrical Vehicle Charging for Demand Response Using Mean-Field Multi-Agent Reinforcement Learning]”** by **[Jiawei You, Yuning Jiang, Xin Liu, Yuanming Shi, Colin N. Jones]**. proposing **[a mean field multi-agent reinforcement learning method for EV charging problem]**. 


## Project Structure

```bash
├── price/                # Data folder (contains price datasets for training and testing)
│   ├── testPrice.xlsx         # Testing data
│   └── trainPrice.xlsx        # Training data
├── src/                  # Source code folder
│   ├── main.py           # Main program
│   ├── utils.py          # Utility functions
│   ├── model.py          # NN Model definition
│   └── ...               # Other scripts
├── run/                  # Results folder (model outputs, visualizations, etc.)
├── README.md             # Project documentation (this file)
├── environment.yml      # Python dependencies
└── LICENSE               # License file
```

## Requirements

Make sure to install the necessary dependencies listed in the `requirements.txt` file. You can install them by running the following command:

```bash
conda env create -f environment.yml
```

### Dependency List:
- Python 3.8
- NumPy >= 1.18.0
- pandas >= 1.0.0
- PyTorch >= 1.5.0 (adjust according to your project)
- Matplotlib >= 3.1.0
- Pypower
- Other dependencies...


## Data 
Our dataset is from ISO New England, the link is https://www.iso-ne.com/isoexpress/

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jiaweiy-kid/MF-SAC.git
   cd your-repository-name
   ```

2. **Install dependencies**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Run the code**:

   - Run the main program:
     ```bash
     python src/main.py
     ```
   - Visualize results:
     ```bash
     tensorboard logdir=run/your result log file
     ```

## Results

The results will be saved in the `run/` directory, including:
- Visualizations (plots, charts)
- Experiment logs

## Contribution

Contributions are welcome! If you have any improvements or suggestions, feel free to submit a pull request, and we will review it promptly.

## License

This project is licensed under the **[License Type, e.g., MIT, Apache 2.0]**. See the [LICENSE](./LICENSE) file for more details.

---

