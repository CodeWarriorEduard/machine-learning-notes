# Deep Learning - Loss Functions Cheat Sheet

## 1. **Regression Loss Functions**
Used for problems where the goal is to predict continuous values.

### **Mean Squared Error (MSE)**
**Usage:** Common in regression problems where we aim to minimize the squared difference between predicted and actual values.  
**Description:** Penalizes large errors more than small ones, making it sensitive to outliers.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

---

### **Mean Absolute Error (MAE)**
**Usage:** Used when you want to penalize errors proportionally to their size.  
**Description:** Unlike MSE, MAE is less sensitive to outliers and calculates the average of the absolute differences between predicted and true values.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

---

### **Huber Loss**
**Usage:** Combines MSE and MAE to be more robust to outliers.  
**Description:** It behaves like MSE for small errors but like MAE for large errors, making it less sensitive to outliers than MSE.

$$
L(y, \hat{y}) =
\begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{if } |y - \hat{y}| > \delta
\end{cases}
$$

---

## 2. **Classification Loss Functions**
Used for classification problems, whether binary or multiclass.

### **Binary Cross-Entropy (Log Loss)**
**Usage:** Common in binary classification problems where we compare the true class labels with the predicted probabilities.  
**Description:** Measures the difference between the true labels and predicted probabilities, penalizing incorrect predictions.

$$
L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

---

### **Categorical Cross-Entropy**
**Usage:** Used for multiclass classification problems where each input can belong to one of many possible classes.  
**Description:** Measures the difference between the true one-hot encoded labels and the predicted probabilities across multiple classes.

$$
L = -\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^C y_{ij} \log(\hat{y}_{ij})
$$

---

### **Sparse Categorical Cross-Entropy**
**Usage:** Similar to categorical cross-entropy, but the labels are integers instead of one-hot encoded vectors.  
**Description:** Used when the class labels are not one-hot encoded but represented as integers.

---

### **Hinge Loss**
**Usage:** Used for Support Vector Machines (SVMs) for binary classification tasks.  
**Description:** Focuses on maximizing the margin between the classes and penalizes incorrect predictions that are too close to the decision boundary.

$$
L = \sum_{i=1}^n \max(0, 1 - y_i \hat{y}_i)
$$

---

### **Focal Loss**
**Usage:** Used in problems with imbalanced classes, focusing more on hard-to-classify examples.  
**Description:** Applies more weight to difficult examples, making it especially useful for imbalanced datasets, such as in object detection.

$$
L = -\alpha (1 - \hat{p}_t)^\gamma \log(\hat{p}_t)
$$

---

## 3. **Segmentation Loss Functions**
Used for tasks where the goal is to classify each pixel of an image.

### **Dice Loss**
**Usage:** Common in medical image segmentation tasks where accuracy of pixel-wise classification is critical.  
**Description:** Measures the overlap between two sets, useful for handling imbalanced data in segmentation tasks.

$$
\text{Dice Loss} = 1 - \frac{2 \cdot |A \cap B|}{|A| + |B|}
$$

---

### **IoU (Intersection over Union) Loss**
**Usage:** Used for evaluating the performance of segmentation models by measuring the overlap between predicted and true regions.  
**Description:** Measures the area of overlap divided by the area of union between the predicted and true masks.

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

---

## 4. **Generative Models Loss Functions**
Used in generative tasks such as GANs and autoencoders.

### **Adversarial Loss (GANs)**
**Usage:** Used in Generative Adversarial Networks (GANs), where a generator tries to create realistic data and a discriminator tries to distinguish between real and fake data.  
**Description:** The generator aims to fool the discriminator, while the discriminator aims to correctly identify real vs fake data.

---

### **Autoencoder Loss (Reconstruction Loss)**
**Usage:** Used in autoencoders to minimize the reconstruction error between the input and output.  
**Description:** Measures the difference between the input and the reconstructed output, typically using MSE or MAE.

$$
L = \sum_{i=1}^n \left( x_i - \hat{x}_i \right)^2
$$

---

## 5. **Reinforcement Learning Loss Functions**
Used for learning tasks in which an agent interacts with an environment and maximizes a reward.

### **Policy Gradient Loss**
**Usage:** Used in reinforcement learning to optimize the policy by adjusting weights to maximize the cumulative reward.  
**Description:** Updates the policy by maximizing the expected reward based on the actions taken.

$$
L = - \sum_{i=1}^n \log(\pi(a_i | s_i)) \cdot R_i
$$

---

### **Q-Learning Loss**
**Usage:** Used in Q-learning, a type of reinforcement learning, to update the Q-values for state-action pairs.  
**Description:** Minimizes the difference between the predicted Q-values and the actual reward, adjusting the agent's decision-making process.

$$
L = (r + \gamma \cdot \max_a Q(s', a) - Q(s, a))^2
$$

---

## 6. **Other Loss Functions**
Used in various specialized tasks.

### **Kullback-Leibler Divergence (KL Divergence)**
**Usage:** Used to measure how one probability distribution diverges from a second, expected probability distribution.  
**Description:** Often used in generative models and variational autoencoders.

$$
D_{KL}(P || Q) = \sum_i P(i) \log\left(\frac{P(i)}{Q(i)}\right)
$$

---

### **Contrastive Loss**
**Usage:** Used in tasks where the model learns to differentiate between similar and dissimilar pairs, such as in metric learning.  
**Description:** Minimizes the distance between similar pairs and maximizes the distance between dissimilar pairs.

$$
L = \frac{1}{2} \cdot y \cdot D^2 + (1 - y) \cdot \max(0, m - D)^2
$$

---

### **Triplet Loss**
**Usage:** Used to learn a distance metric between similar and dissimilar examples, typically in face recognition or verification tasks.  
**Description:** Minimizes the distance between an anchor and a positive example, while maximizing the distance between the anchor and a negative example.

$$
L = \max(d(a, p) - d(a, n) + \alpha, 0)
$$

