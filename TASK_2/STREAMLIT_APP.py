import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import MNISTNet
import os


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)

    if os.path.exists('mnist_cnn.pth'):
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
        model.eval()
        return model, device, True
    else:
        return model, device, False


@st.cache_data
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    return test_dataset


def evaluate_model(model, device):
    test_dataset = load_test_data()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def predict_samples(model, device, num_samples=5):
    test_dataset = load_test_data()

    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for idx, ax in enumerate(axes):
        image, label = test_dataset[indices[idx]]

        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0][predicted_class].item() * 100

        img_np = image.squeeze().numpy()

        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'True: {label}\nPred: {predicted_class}\nConf: {confidence:.1f}%',
                     color='green' if predicted_class == label else 'red',
                     fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

    st.title("🔢 MNIST Handwritten Digit Classifier")
    st.markdown("**A Convolutional Neural Network for classifying handwritten digits**")

    model, device, model_exists = load_model()

    if not model_exists:
        st.warning("⚠️ No trained model found. Please train the model first.")
        st.markdown("### Training Instructions")
        st.code("python train.py", language="bash")
        st.info("Run the above command to train the model. This will take a few minutes.")
        return

    st.success("✅ Model loaded successfully!")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📊 Model Information")
        st.markdown(f"**Device:** {device}")
        st.markdown("**Architecture:** CNN with 3 Conv Layers")
        st.markdown("**Input:** 28x28 grayscale images")
        st.markdown("**Output:** 10 classes (digits 0-9)")

        if st.button("🧪 Evaluate Model on Test Set", use_container_width=True):
            with st.spinner("Evaluating model..."):
                accuracy = evaluate_model(model, device)
                st.session_state['accuracy'] = accuracy

        if 'accuracy' in st.session_state:
            acc = st.session_state['accuracy']
            if acc >= 95:
                st.success(f"**Test Accuracy:** {acc:.2f}% ✅")
            else:
                st.warning(f"**Test Accuracy:** {acc:.2f}%")

    with col2:
        st.markdown("### 🖼️ Sample Predictions")

        num_samples = st.slider("Number of samples to visualize", 3, 10, 5)

        if st.button("🎲 Generate New Predictions", use_container_width=True):
            with st.spinner("Generating predictions..."):
                fig = predict_samples(model, device, num_samples)
                st.pyplot(fig)
                plt.close()

    st.markdown("---")

    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Architecture:**
        - **Layer 1:** Conv2D (1→32 filters) + ReLU + MaxPool
        - **Layer 2:** Conv2D (32→64 filters) + ReLU + MaxPool
        - **Layer 3:** Conv2D (64→128 filters) + ReLU + MaxPool
        - **Dropout:** 0.25 and 0.5 for regularization
        - **FC Layer 1:** 128×3×3 → 256 neurons + ReLU
        - **FC Layer 2:** 256 → 10 classes (softmax)

        **Training Details:**
        - **Optimizer:** Adam (lr=0.001)
        - **Loss Function:** CrossEntropyLoss
        - **Epochs:** 10
        - **Batch Size:** 128
        """)

    st.markdown("---")
    st.markdown("*Built with PyTorch and Streamlit*")


if __name__ == "__main__":
    main()
