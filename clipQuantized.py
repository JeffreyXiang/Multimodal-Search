import torch
import torch.nn as nn
import clip


class ClipQuantized(nn.Module):
    def __init__(self, clip_model_path, device, alpha=1, beta=1, M=32, k=256, D=512):
        super(ClipQuantized, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self._M = M  # num of subspace
        self._k = k  # centroids in each subspace
        self._D = D  # vector length
        self._d = D // M  # size of each subspace
        self.clip_model, self.preprocess = clip.load(clip_model_path, device)
        self.idx_base = torch.tensor([self._k * i for i in range(self._M)], dtype=torch.int64).cuda(device)
        #         centroids = nn.init.xavier_uniform_(torch.empty(size=(M, k, D//M), dtype=torch.float32))
        centroids = nn.init.xavier_uniform_(torch.empty(size=(M, k, D//M), dtype=torch.float16))
        self.centroids = nn.Parameter(centroids)
        
    def set_centroids(self, centroids):
        centroids = torch.tensor(centroids, dtype=torch.float16).cuda(self.device)
        self.centroids = nn.Parameter(centroids)
        
    def get_preprocess(self):
        return self.preprocess

    def encode_image(self, image=None):
        image_vec = self.clip_model.encode_image(image).reshape((-1, self._M, self._d))     # (bs, M, d)
        # image_vec = torch.rand((5, self._M, self._d), dtype=torch.float32)
        image_norm = torch.sum(image_vec ** 2, dim=2).unsqueeze(2)              # (bs, M, 1)
        centroids_norm = torch.sum(self.centroids ** 2, dim=2).unsqueeze(0)     # (1, M, k)
        image_dot = torch.matmul(self.centroids,
                                 image_vec.unsqueeze(-1)).squeeze()         # (M, k, d)x(bs, M, d, 1) -> (bs, M, k)
        image_dist = image_norm + centroids_norm - 2 * image_dot            # (bs, M, k)
        image_quantized = torch.argmin(image_dist, dim=2, keepdim=False)  # (bs, M)

        
        image_quantized += self.idx_base.unsqueeze(0)  # (bs, M)

        centroids = self.centroids.reshape((-1, self._d))   # (M * k, d)
        image_quantized = image_quantized.reshape(-1)       # (bs * M)

        output = torch.index_select(centroids, 0, image_quantized)  # (bs * M, d)
        output = output.reshape((-1, self._D))            # (bs, M * d)
        image_vec = image_vec.reshape((-1, self._D))      # (bs, M * d)
        quantize_loss = torch.mean(torch.sum((output - image_vec.detach())**2, -1)) * self.alpha + torch.mean(torch.sum((output.detach() - image_vec)**2, -1)) * self.beta
        # print(image_vec[0, ...], output[0, ...])

        return (output - image_vec).detach() + image_vec, quantize_loss

    def encode_text(self, text=None):
        text_vec = self.clip_model.encode_text(text).reshape((-1, self._M, self._d))     # (bs, M, d)
        # text_vec = torch.rand((100, self._M, self._d), dtype=torch.float32)
        text_norm = torch.sum(text_vec ** 2, dim=2).unsqueeze(2)  # (bs, M, 1)
        centroids_norm = torch.sum(self.centroids ** 2, dim=2).unsqueeze(0)  # (1, M, k)
        text_dot = torch.matmul(self.centroids,
                                text_vec.unsqueeze(-1)).squeeze()  # (M, k, d)x(bs, M, d, 1) -> (bs, M, k)
        text_dist = text_norm + centroids_norm - 2 * text_dot  # (bs, M, k)
        text_quantized = torch.argmin(text_dist, dim=2, keepdim=False)  # (bs, M)

        text_quantized += self.idx_base.unsqueeze(0)  # (bs, M)

        centroids = self.centroids.reshape((-1, self._d))  # (M * k, d)
        text_quantized = text_quantized.reshape(-1)  # (bs * M)

        output = torch.index_select(centroids, 0, text_quantized)  # (bs * M, d)
        output = output.reshape((-1, self._D))  # (bs, M * d)
        text_vec = text_vec.reshape((-1, self._D))  # (bs, M * d)
        quantize_loss = torch.mean(torch.sum((output - text_vec.detach())**2, -1)) * self.alpha + torch.mean(torch.sum((output.detach() - text_vec)**2, -1)) * self.beta
        # print(text_vec[0, ...], output[0, ...])

        return (output - text_vec).detach() + text_vec, quantize_loss

    def forward(self, image, text):
        image_vec, image_quant_loss = self.encode_image(image)
        text_vec, text_quant_loss = self.encode_text(text)
        quant_loss = (image_quant_loss + text_quant_loss)
        similarity = (100.0 * image_vec @ text_vec.T).softmax(dim=-1)

        return similarity, quant_loss

    
class ClipQuantizedI2T(ClipQuantized):
    def __init__(self, clip_model_path, device, alpha=1, beta=1, M=32, k=256, D=512):
        super(ClipQuantizedI2T, self).__init__(clip_model_path, device, alpha, beta, M, k, D)
    
    def encode_image(self, image=None):
        output = self.clip_model.encode_image(image)
        return output, 0


class ClipQuantizedT2I(ClipQuantized):
    def __init__(self, clip_model_path, device, alpha=1, beta=1, M=32, k=256, D=512):
        super(ClipQuantizedT2I, self).__init__(clip_model_path, device, alpha, beta, M, k, D)

    def encode_text(self, text=None):
        output = self.clip_model.encode_text(text)
        return output, 0
    

if __name__ == '__main__':
    model, preprocess = clip.load("notebooks/model.pt", "cuda:0")
    a = ClipQuantized(model)
    for name, param in a.named_parameters():
        print(name)
    a.encode_image()

