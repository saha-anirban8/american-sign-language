import torch
from torchvision import transforms, models
import torch.nn as nn
import PIL


class VGG16Model:

    def __init__(
            self,
            path_to_pretrained_model: str = None
            ):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        if path_to_pretrained_model:
            self.model = models.vgg16(pretrained=False)
            self.model.classifier[6] = nn.Linear(in_features=4096, out_features=29, bias=True)
            self.model.load_state_dict(torch.load(path_to_pretrained_model, map_location='cpu'))

        self.data_transform = self._setup_transform()

    def predict_proba(
            self,
            img: PIL.Image.Image,
            k: int,
            index_to_class_labels: dict,
            show: bool = False
            ):
        if show:
            img.show()
        img = self.data_transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        self.model.eval()
        output_tensor = self.model(img)
        prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
        top_k = torch.topk(prob_tensor, k, dim=1)
        probabilites = top_k.values.detach().numpy().flatten()
        indices = top_k.indices.detach().numpy().flatten()
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_label = index_to_class_labels[pred_idx].title()
            predicted_perc = pred_prob * 100
            formatted_predictions.append(
                (predicted_label, f"{predicted_perc:.3f}%"))

        return formatted_predictions

    def _setup_transform(self):

        data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.ToTensor()])

        return data_transform
