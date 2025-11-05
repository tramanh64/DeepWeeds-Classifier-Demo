# 🌿 DeepWeeds - AI Weed Identification (EfficientNetB3)

[cite_start]A Flask web application that uses a fine-tuned EfficientNetB3 model to classify 9 classes of Australian weeds (8 weed species and 1 negative class) from the DeepWeeds dataset[cite: 15, 205, 212].

[cite_start]This project was developed as the final report for the "Deep Learning Theory and Applications" course at Ho Chi Minh City University of Technology and Education[cite: 7, 12].

## ✨ Features

* [cite_start]**Image Upload**: Simple UI to upload an image for classification[cite: 355].
* [cite_start]**AI Recognition**: Analyzes the image using a trained EfficientNetB3 model[cite: 306, 355].
* [cite_start]**Bilingual Results**: Returns the weed's name in both English and Vietnamese[cite: 417].
* [cite_start]**Confidence Score**: Displays the model's prediction confidence percentage[cite: 419].

---

## 🚀 Tech Stack

* [cite_start]**Backend**: Flask [cite: 358]
* [cite_start]**Deep Learning**: TensorFlow 2.x, Keras [cite: 270]
* [cite_start]**Frontend**: HTML, CSS (for the demo interface) [cite: 360]
* [cite_start]**Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib [cite: 274, 275, 277, 278]

---

## 🤖 Model Details

### 1. Dataset

* [cite_start]**Name**: [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) [cite: 205, 206]
* [cite_start]**Subset Used**: The model was trained on `train_subset0.csv` (10,501 images) and validated on `val_subset0.csv` (3,501 images)[cite: 227, 228, 274].
* [cite_start]**Classes**: 9 total classes (Labels 0-8)[cite: 212, 283]:
    1.  [cite_start]`0`: Chinee apple [cite: 285]
    2.  [cite_start]`1`: Lantana [cite: 286]
    3.  [cite_start]`2`: Parkinsonia [cite: 287]
    4.  [cite_start]`3`: Parthenium [cite: 288]
    5.  [cite_start]`4`: Prickly acacia [cite: 290]
    6.  [cite_start]`5`: Rubber vine [cite: 291]
    7.  [cite_start]`6`: Siam weed [cite: 292]
    8.  [cite_start]`7`: Snake weed [cite: 293]
    9.  [cite_start]`8`: Negative (background) [cite: 294]

### 2. Architecture

* [cite_start]**Base Model**: `EfficientNetB3` pre-trained on ImageNet, with the top classification layer removed[cite: 306, 308].
* [cite_start]**Custom Head**: A custom classifier was added on top of the base model[cite: 309, 343]:
    ```
    [cite_start]EfficientNetB3 Base [cite: 340]
            |
    [cite_start]GlobalAveragePooling2D() [cite: 310, 342]
            |
    [cite_start]Dense(512, activation='relu') [cite: 311, 344]
            |
    [cite_start]Dropout(0.5) [cite: 312, 346]
            |
    [cite_start]Dense(9, activation='softmax') [cite: 315, 347]
    ```

### 3. Training Strategy

[cite_start]A 3-stage fine-tuning strategy was employed for optimal performance[cite: 324]:

1.  [cite_start]**Stage 1: Warm-up (15 epochs)** [cite: 326]
    * [cite_start]**Frozen Layers**: Entire `base_model` (EfficientNetB3) was frozen[cite: 328].
    * [cite_start]**Goal**: Train only the new custom head[cite: 327].
    * [cite_start]**Learning Rate**: `1e-4`[cite: 328].

2.  [cite_start]**Stage 2: Shallow Fine-tuning (Epochs 15-35)** [cite: 329]
    * [cite_start]**Unfrozen Layers**: `block6` and `block7` were unfrozen[cite: 331].
    * [cite_start]**Goal**: Fine-tune the high-level feature blocks[cite: 330].
    * [cite_start]**Learning Rate**: `CosineDecay` starting from `1e-5` (1e-4 / 10)[cite: 332].

3.  [cite_start]**Stage 3: Deep Fine-tuning (Epochs 35-50)** [cite: 334]
    * [cite_start]**Unfrozen Layers**: `block4` and `block5` were also unfrozen[cite: 336].
    * [cite_start]**Goal**: Fine-tune deeper feature blocks[cite: 335].
    * [cite_start]**Learning Rate**: `CosineDecay` starting from `2e-6` (1e-4 / 50)[cite: 337].

### 4. Performance

* [cite_start]**Final Accuracy**: **83%** (Overall Accuracy)[cite: 462, 479].
* [cite_start]**Evaluation Method**: Achieved using **Test-Time Augmentation (TTA)**, where predictions were averaged over 5 augmented versions of each validation image[cite: 458, 459].
* **Key Results**:
    * [cite_start]Excellent F1-scores for **Class 2 (Parkinsonia)** and **Class 5 (Rubber Vine)** (both 0.85)[cite: 486].
    * [cite_start]High recall (0.94) for the majority **Class 8 (Negative)**, showing the model is good at identifying non-weeds[cite: 483].
    * [cite_start]The model struggled with **Class 3 (Parthenium)** (Recall: 0.47) [cite: 488][cite_start], which was often confused with Class 4 and Class 8, indicating visual similarity and a challenge from data imbalance[cite: 595, 596, 598].

---

## 🔧 How to Run the Application

(This section answers the second part of your request)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)
    cd YOUR_REPO
    ```

2.  **Prepare Files:**
    * [cite_start]Make sure your directory structure matches the one in the report [cite: 365-371]:
    ```
    DeepWeeds_Demo/
    ├── app.py
    [cite_start]├── deepweeds_effnet_final.keras  <-- IMPORTANT: Place your trained model file here! [cite: 367]
    ├── requirements.txt              <-- Create this file (see step 4)
    ├── templates/
    │   ├── index.html
    │   └── result.html
    └── uploads/                        <-- This folder will be created by app.py
    ```

3.  **Create a Virtual Environment:**
    ```bash
    # Create a new environment named 'venv'
    python -m venv venv

    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Dependencies:**
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

5.  **Run the Flask App:**
    * In your terminal (with the virtual environment still active), run:
    ```bash
    flask run
    ```
    * [cite_start]Open your web browser and go to: `http://127.0.0.1:5000`[cite: 399].

You can now upload an image and test your model!