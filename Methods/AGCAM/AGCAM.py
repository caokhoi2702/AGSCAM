import torch
import torchvision.transforms as transforms
from einops.layers.torch import Reduce, Rearrange
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class BetterAGCAM:
    """Implementation of our method."""

    def __init__(
        self,
        model,
        attention_matrix_layer="before_softmax",
        attention_grad_layer="after_softmax",
        head_fusion="sum",
        layer_fusion="sum",
    ):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)

    def get_attn_matrix(self, module, input, output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.attn_matrix.append(
            output[:, :, 0:1, :]
        )  # shape: [batch, num_heads, 1, num_patches]

    def get_grad_attn(self, module, grad_input, grad_output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.grad_attn.append(
            grad_output[0][:, :, 0:1, :]
        )  # shape: [batch, num_heads, 1, num_patches]

    def generate(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        # backpropagate the model from the classification output
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction
        if cls_idx == None:  # generate CAM for a certain class label
            loss = output[0, prediction[0]]
        else:  # generate CAM for the predicted class
            loss = output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        self.head = h
        self.width = int((d - 1) ** 0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        gradient = self.grad_attn[0]
        for i in range(1, len(self.attn_matrix)):
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(
            gradient
        )  # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(
            attn
        )  # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        heatmaps = Rearrange(
            "b l hd z (h w)  -> b l hd z h w", h=self.width, w=self.width
        )(mask[:, :, :, 1:].unsqueeze(0))
        
        sum_heatmap = []
        
        for i in range(12):
            for j in range(12):
                
                heatmap = heatmaps[0][i][j]
                up_heatmap = heatmap.reshape(1, 1, 14, 14)
                upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
                up_heatmap = upsample(up_heatmap)
                norm_heatmap = (up_heatmap - up_heatmap.min())/(up_heatmap.max()-up_heatmap.min())
                new_image = input_tensor * norm_heatmap
                masked_output = self.model.__call__(new_image) 
                conf = masked_output - output
                conf = conf[0, prediction.item()].detach().cpu()

                # Generate new heatap
                sum_heatmap.append(conf)

        # Calculate alpha using softmax to get the contribution of each heatmap
        sum_heatmap = np.array(sum_heatmap, dtype=float)
        sum_heatmap = torch.tensor(sum_heatmap, dtype=torch.float)
        sigmoid_alpha = torch.sigmoid(sum_heatmap)
        sigmoid_alpha = sigmoid_alpha.unsqueeze(1).unsqueeze(2).repeat(1, 1, 196)
        sigmoid_heatmap = sigmoid_alpha * heatmaps.reshape(144, 1, 196).detach().cpu()
        sigmoid_heatmap = torch.sum(sigmoid_heatmap, axis = 0)

        
        sigmoid_heatmap = sigmoid_heatmap.reshape(1, 1, 14, 14)
        sigmoid_heatmap = transforms.Resize((224, 224))(sigmoid_heatmap[0])
        sigmoid_heatmap = (sigmoid_heatmap - sigmoid_heatmap.min())/(sigmoid_heatmap.max()-sigmoid_heatmap.min())
        sigmoid_heatmap = sigmoid_heatmap.detach().cpu().numpy()
        sigmoid_heatmap = np.transpose(sigmoid_heatmap, (1, 2, 0))
        # tensor_heatmaps = a_mask[0]
        # tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
        # tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)

        # min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)
        # max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)

        # tensor_heatmaps = (tensor_heatmaps - min_vals + 1e-7) / (
        #     max_vals - min_vals + 1e-7
        # )

        # m = torch.mul(tensor_heatmaps, input_tensor)
        # with torch.no_grad():
        #     output_mask = self.model.__call__(m)

        # agc_scores = output_mask[:, prediction.item()] - output[0, prediction.item()]
        # # print('score shape: ', agc_scores.shape)
        # agc_scores = torch.sigmoid(agc_scores).reshape(12, 12)
        # my_cam = (agc_scores.view(12, 12, 1, 1, 1) * a_mask[0]).sum(axis=(0, 1))
        # # my_cam = (agc_scores[:, :, None, None, None] * a_mask[0]).sum(axis=(0, 1))

        # # sigmoid_mask = torch.from_numpy(my_cam)
        # sigmoid_mask = my_cam.unsqueeze(0)

        return prediction, sigmoid_heatmap


class AGCAM:
    """Implementation of our method."""

    def __init__(
        self,
        model,
        attention_matrix_layer="before_softmax",
        attention_grad_layer="after_softmax",
        head_fusion="sum",
        layer_fusion="sum",
    ):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)

    def get_attn_matrix(self, module, input, output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.attn_matrix.append(
            output[:, :, 0:1, :]
        )  # shape: [batch, num_heads, 1, num_patches]

    def get_grad_attn(self, module, grad_input, grad_output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.grad_attn.append(
            grad_output[0][:, :, 0:1, :]
        )  # shape: [batch, num_heads, 1, num_patches]

    def generate(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        # backpropagate the model from the classification output
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction
        if cls_idx == None:  # generate CAM for a certain class label
            loss = output[0, prediction[0]]
        else:  # generate CAM for the predicted class
            loss = output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        self.head = h
        self.width = int((d - 1) ** 0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        gradient = self.grad_attn[0]
        for i in range(1, len(self.attn_matrix)):
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(
            gradient
        )  # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(
            attn
        )  # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        mask = mask[:, :, :, 1:].unsqueeze(0)
        mask = Reduce("b l h z p -> b l z p", reduction=self.head_fusion)(mask)
        mask = Reduce("b l z p -> b z p", reduction=self.layer_fusion)(mask)
        mask = Rearrange("b z (h w) -> b z h w", h=self.width, w=self.width)(mask)

        return prediction, mask


class OptiAGCAM:
    """Implementation of our method."""

    def __init__(
        self,
        model,
        attention_matrix_layer="before_softmax",
        attention_grad_layer="after_softmax",
        head_fusion="sum",
        layer_fusion="sum",
    ):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)

    def get_attn_matrix(self, module, input, output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.attn_matrix.append(
            output[:, :, 0:1, :]
        )  # shape: [batch, num_heads, 1, num_patches]

    def get_grad_attn(self, module, grad_input, grad_output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.grad_attn.append(
            grad_output[0][:, :, 0:1, :]
        )  # shape: [batch, num_heads, 1, num_patches]

    def generate(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        # backpropagate the model from the classification output
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction
        if cls_idx == None:  # generate CAM for a certain class label
            loss = output[0, prediction[0]]
        else:  # generate CAM for the predicted class
            loss = output[0, cls_idx]
        loss.backward(retain_graph=True)

        b, h, n, d = self.attn_matrix[0].shape
        self.head = h
        self.width = int((d - 1) ** 0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        gradient = self.grad_attn[0]
        for i in range(1, len(self.attn_matrix)):
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(
            gradient
        )  # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(
            attn
        )  # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        mask = mask[:, :, :, 1:]

        mask = mask.reshape(144, 1, 196)

        # mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
        # mask = Reduce('b l z p -> b z p', reduction=self.layer_fusion)(mask)
        # mask = Rearrange('b z (h w) -> b z h w', h=self.width, w=self.width)(mask)
        return prediction, mask, output
