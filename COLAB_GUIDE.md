# How to Run SSCLNet in Google Colab

Since you want to move from Local to Cloud (Colab) to utilize faster GPUs, follow these steps:

## 1. Prepare your Project
1.  **Zip the Folder**: Compress your entire `SSCLNet_Project` folder into a zip file named `SSCLNet_Project.zip`.
    - Make sure the `datasets` folder is inside!

## 2. Google Colab Setup
1.  Open [Google Colab](https://colab.research.google.com/).
2.  **Upload Notebook**: Upload the `SSCLNet_Runner.ipynb` file (I just created this for you in the project folder).
3.  **Enable GPU**: Go to `Runtime` > `Change runtime type` > Select **T4 GPU**.

## 3. Upload Data
1.  In the Colab sidebar (Files icon), click the **Upload** button.
2.  Select your `SSCLNet_Project.zip` file.
3.  *Wait for the upload to complete (blue circle fills up).*

## 4. Run the Notebook
1.  Run the **Setup** cell to unzip your project.
2.  Run the **Install Dependencies** cell.
3.  Run the **Training** cells.
    - *Note: It will be MUCH faster on Colab GPU than your local CPU!*

## 5. View the App
1.  Run the final **Streamlit** cell.
2.  It will provide a URL (e.g., `your-url.loca.lt`).
3.  Click it to open your app!
    - *Note: It might ask for a password/IP. The IP is usually displayed in the cell output or you can get it by running `!curl ipv4.icanhazip.com` in a new cell.*
