# 🌿 DeepWeeds - AI Weed Identification (EfficientNetB3)

A Flask web application that uses a fine-tuned EfficientNetB3 model to classify 9 classes of Australian weeds (8 weed species and 1 negative class) from the DeepWeeds dataset.

This project was developed as the final report for the "Deep Learning Theory and Applications" course at Ho Chi Minh City University of Technology and Education.

##  Features
* **Image Upload**: Simple UI to upload an image for classification.
* **AI Recognition**: Analyzes the image using a trained EfficientNetB3 model.
* **Bilingual Results**: Returns the weed's name in both English and Vietnamese.
* **Confidence Score**: Displays the model's prediction confidence percentage.

## 📸 Demo
Below are screenshots of the application in action:

### Upload Page

![z7193506890122_d90459d12050959b393a53ffc2428400](https://github.com/user-attachments/assets/03c99bd4-e1be-4d4b-a6fb-021049d13dd7)

### Result Page
![z7193506891078_705ac611084dee3fa08c5a2e8e3c57c6](https://github.com/user-attachments/assets/dea78fc0-3881-4150-9e59-0985e2a7cc6b)

Kaggle Notebook
The complete model training, fine-tuning, and evaluation process was conducted in a Kaggle notebook.

You can view the full notebook here: [DeepWeeds - EfficientNetB3 Training](https://www.kaggle.com/code/trmanhinhngc/deepweeds-efficientnetb3?scriptVersionId=268674068)

## Tech Stack
* **Backend**: Flask
* **Deep Learning**: TensorFlow 2.x, Keras
* **Frontend**: HTML, CSS (for the demo interface)
* **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib

## Model Details
### 1. Dataset
* **Name**: [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds)
* **Subset Used**: The model was trained on `train_subset0.csv` (10,501 images) and validated on `val_subset0.csv` (3,501 images).
* **Classes**: 9 total classes (Labels 0-8):
    1. `0`: Chinee apple
    2. `1`: Lantana
    3. `2`: Parkinsonia
    4. `3`: Parthenium
    5. `4`: Prickly acacia
    6. `5`: Rubber vine
    7. `6`: Siam weed
    8. `7`: Snake weed
    9. `8`: Negative (background)

### 2. Architecture
* **Base Model**: `EfficientNetB3` pre-trained on ImageNet, with the top classification layer removed.
* **Custom Head**: A custom classifier was added on top of the base model:
    ```
    EfficientNetB3 Base
            |
    GlobalAveragePooling2D()
            |
    Dense(512, activation='relu')
            |
    Dropout(0.5)
            |
    Dense(9, activation='softmax')
    ```

### 3. Training Strategy
A 3-stage fine-tuning strategy was employed for optimal performance:
1. **Stage 1: Warm-up (15 epochs)**
    * **Frozen Layers**: Entire `base_model` (EfficientNetB3) was frozen.
    * **Goal**: Train only the new custom head.
    * **Learning Rate**: `1e-4`.
2. **Stage 2: Shallow Fine-tuning (Epochs 15-35)**
    * **Unfrozen Layers**: `block6` and `block7` were unfrozen.
    * **Goal**: Fine-tune the high-level feature blocks.
    * **Learning Rate**: `CosineDecay` starting from `1e-5` (1e-4 / 10).
3. **Stage 3: Deep Fine-tuning (Epochs 35-50)**
    * **Unfrozen Layers**: `block4` and `block5` were also unfrozen.
    * **Goal**: Fine-tune deeper feature blocks.
    * **Learning Rate**: `CosineDecay` starting from `2e-6` (1e-4 / 50).

### 4. Performance
* **Final Accuracy**: **83%** (Overall Accuracy).
* **Evaluation Method**: Achieved using **Test-Time Augmentation (TTA)**, where predictions were averaged over 5 augmented versions of each validation image.
* **Key Results**:
    * Excellent F1-scores for **Class 2 (Parkinsonia)** and **Class 5 (Rubber Vine)** (both 0.85).
    * High recall (0.94) for the majority **Class 8 (Negative)**, showing the model is good at identifying non-weeds.
    * The model struggled with **Class 3 (Parthenium)** (Recall: 0.47), which was often confused with Class 4 and Class 8, indicating visual similarity and a challenge from data imbalance.

## 🔧 How to Run the Application
1. **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO
    ```
2. **Prepare Files:**
    * Make sure your directory structure matches the one in the report:
    ```
    DeepWeeds_Demo/
    ├── app.py
    ├── deepweeds_effnet_final.keras <-- IMPORTANT: Place your trained model file here!
    ├── requirements.txt <-- Create this file (see step 4)
    ├── templates/
    │ ├── index.html
    │ └── result.html
    └── uploads/ <-- This folder will be created by app.py
    ```
3. **Create a Virtual Environment:**
    ```bash
    # Create a new environment named 'venv'
    python -m venv venv
    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
4. **Install Dependencies:**
    * Create a file named `requirements.txt` in your project folder.
    * Add the following lines to it:
    ```
    flask
    tensorflow
    numpy
    ```
    * Install these libraries by running:
    ```bash
    pip install -r requirements.txt
    ```
5. **Run the Flask App:**
    * In your terminal (with the virtual environment still active), run:
    ```bash
    flask run
    ```
    * Open your web browser and go to: `http://127.0.0.1:5000`.
You can now upload an image and test your model!
