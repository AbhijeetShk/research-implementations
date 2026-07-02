import torch


def get_output_dim(
    model,
    input_length=200
):
    with torch.no_grad():
        x = torch.randn(
            1,
            1,
            input_length
        )

        output = model(x)

    return output.shape[1]