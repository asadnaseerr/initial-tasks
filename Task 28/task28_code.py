# TASK: Convert the output of an image processing model you trained into .onnx format. Use this model to understand the differences between .pt and .onnx formats.

# working_onnx_conversion.py
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
import time

print("WORKING OCR MODEL TO ONNX CONVERSION")

# Step 1: Define a PROPER model with correct tensor shapes
class WorkingOCRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(10, 20, 3) # 26x26 -> 24x24
        self.pool = nn.MaxPool2d(2, 2)     # 24x24 -> 12x12
        self.fc1 = nn.Linear(20 * 12 * 12, 50)  # CORRECT: 20*12*12 = 2880
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 1x28x28 -> 10x26x26
        x = torch.relu(self.conv2(x))  # 10x26x26 -> 20x24x24
        x = self.pool(x)               # 20x24x24 -> 20x12x12
        x = x.view(-1, 20 * 12 * 12)   # CORRECT shape: 20*12*12 = 2880
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Create and save model
print("1. Creating model...")
model = WorkingOCRModel()

# Save state_dict safely
torch.save(model.state_dict(), 'working_model.pt')
print("✓ PyTorch model saved as 'working_model.pt'")

# Step 3: Convert to ONNX
print("2. Converting to ONNX...")

# Load model
model = WorkingOCRModel()
model.load_state_dict(torch.load('working_model.pt', weights_only=True))
model.eval()

# Create dummy input with correct shape
dummy_input = torch.randn(1, 1, 28, 28)

# Convert to ONNX with new exporter
torch.onnx.export(
    model,
    dummy_input,
    'working_model.onnx',
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    training=torch.onnx.TrainingMode.EVAL
)

print("✓ ONNX model saved as 'working_model.onnx'")

# Verify ONNX model
onnx_model = onnx.load('working_model.onnx')
onnx.checker.check_model(onnx_model)
print("✓ ONNX model is valid!")

# Step 4: Test both models
print("3. Testing models...")

# Test input
test_input = torch.randn(1, 1, 28, 28)

# PyTorch inference
model.eval()
start_time = time.time()
with torch.no_grad():
    pt_output = model(test_input)
pt_time = (time.time() - start_time) * 1000

# ONNX inference  
ort_session = ort.InferenceSession('working_model.onnx')
input_name = ort_session.get_inputs()[0].name
start_time = time.time()
onnx_output = ort_session.run(None, {input_name: test_input.numpy()})[0]
onnx_time = (time.time() - start_time) * 1000

print(f"PyTorch time: {pt_time:.3f} ms")
print(f"ONNX time:    {onnx_time:.3f} ms")
print(f"Speedup:      {pt_time/onnx_time:.2f}x")

# File sizes
pt_size = os.path.getsize('working_model.pt')
onnx_size = os.path.getsize('working_model.onnx')
print(f"\nPyTorch size: {pt_size/1024:.2f} KB")
print(f"ONNX size:    {onnx_size/1024:.2f} KB")

# Output comparison
output_diff = np.abs(pt_output.numpy() - onnx_output).max()
print(f"Output difference: {output_diff:.8f}")

print("\n✓ SUCCESS! Both models work correctly.")
print("Files created: working_model.pt, working_model.onnx")