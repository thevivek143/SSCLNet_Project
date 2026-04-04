
import matplotlib.pyplot as plt
import pandas as pd

def create_table_image():
    # Data provided by user
    data = [
        ["Brain MRI", "SSCLNet (SimCLR + DenseNet Encoder)", "97.82", "0.978", "0.978", "0.978"],
        ["Knee X-ray", "DenseNet121 + SE-Attention (SSCLNet)", "74.21", "0.69", "0.78", "0.71"]
    ]
    columns = ["Modality", "Model Architecture", "Accuracy (%)", "Precision", "Recall", "F1-Score"]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 3)) # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5) # Scale width and height

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # Dark blue-ish header
        else:
            # cell.set_facecolor('#f5f5f5')
            pass
            
        # Bold specific "winning" model names or similar if desired, but keeping simple clean style

    # Save
    output_path = "SSCLNet_Performance_Table.png"
    plt.title("SSCLNet Performance Metrics", fontsize=16, weight='bold', y=0.85)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Table saved to {output_path}")

if __name__ == "__main__":
    create_table_image()
