import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1

class InceptionResnetV1_128d(nn.Module):
    """
    A modified InceptionResnetV1 model that outputs 128-dimensional embeddings.

    This model loads the pretrained VGGFace2 weights for InceptionResnetV1
    and replaces the final layers to produce 128-d features instead of 512-d.
    """
    def __init__(self, pretrained='vggface2', dropout_prob=0.6):
        super().__init__()

        # Load the original pretrained model
        self.backbone = InceptionResnetV1(
            pretrained=pretrained,
            dropout_prob=dropout_prob
        )

        # --- Replace the final layers ---
        # The original model has:
        # self.last_linear = nn.Linear(1792, 512, bias=False)
        # self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        # We'll replace them with versions that output 128 features.

        # The input feature size to the last linear layer is 1792
        in_features = self.backbone.last_linear.in_features

        # New linear layer for 128-d embedding
        self.backbone.last_linear = nn.Linear(in_features, 128, bias=False)

        # New batch norm layer
        self.backbone.last_bn = nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)

        print("--- Modified InceptionResnetV1 ---")
        print("Output embedding dimension: 128")
        print("Pretrained weights loaded from: '{}'".format(pretrained))
        print("---------------------------------")

    def forward(self, x):
        """
        Calculates the 128-dimensional embedding for a given batch of images.
        The output is L2-normalized, same as the original model.
        """
        return self.backbone(x)

def get_finetune_model(dropout_prob=0.6):
    """
    Helper function to create the modified model for finetuning.
    """
    model = InceptionResnetV1_128d(
        pretrained='vggface2',
        dropout_prob=dropout_prob
    )
    return model

if __name__ == '__main__':
    # Example of creating the model and doing a forward pass

    # Create the model
    model = get_finetune_model()
    model.eval() # Set to evaluation mode

    # Create a dummy input tensor (batch_size=2, channels=3, height=160, width=160)
    # The standard input size for facenet is 160x160
    dummy_input = torch.randn(2, 3, 160, 160)

    # Get the output embeddings
    with torch.no_grad():
        embeddings = model(dummy_input)

    # Check the output shape
    print(f"\n--- Model Test ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output embedding shape: {embeddings.shape}")

    # Verify the output dimension is 128
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 128

    print("\nModel created and tested successfully!")
