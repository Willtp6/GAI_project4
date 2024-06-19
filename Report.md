# Report
>GAI project 4 

## 採用方法

使用說明文件中的第一種方法，Accelerating DDPM with DIP-based Initial Priors，主要分為DIP、DDPM兩大部分


### Deep Image Prior
根據[參考資料](https://youtu.be/FPzi8cUhNNY?si=8n8xqAq4LTwlreLy)實做一個CNN做圖像處理，相較於資料中的內容，採用之圖片大小為**256*256**採用較少的convolution layer可以減少計算的複雜程度，加速整體的訓練過程，同時將其修改為可以使用RGB圖片的輸入。
```python=
class CNN(nn.Module):
  def __init__(self, n_lay, n_chan, ksize):
    super(CNN, self).__init__()
    pd = int(ksize/2)
    layers = [nn.Conv2d(3, n_chan, ksize, padding=pd), nn.ReLU(),]
    for _ in range(n_lay):
      layers.append(nn.Conv2d(n_chan, n_chan, ksize, padding=pd))
      layers.append(nn.ReLU())
    layers.append(nn.Conv2d(n_chan, 3, ksize, padding=pd)) 
    layers.append(nn.ReLU())

    self.deep_net = nn.Sequential(*layers)

  def forward(self, x):
    return torch.squeeze(self.deep_net(x))
```


![output-onlinepngtools](https://hackmd.io/_uploads/BJuJWreIC.png)
左上為訓練過程loss下降得到的折線圖，中上為加入雜訊後的原圖，右上為原圖，左下為初始之輸入，為亂數生成的隨機圖片，中下為目前epoch輸出，右下為最佳輸出。
預設在loss過低之前將其停下，原因為此CNN並非為了要得到精準圖片，並且過多的訓練次數造成過擬合會導致一併將雜訊輸出，(如折線圖所示，約3500epoch的交叉代表了過擬合發生，模型開始將雜訊輸出)，訓練過程中將每10 epoch所得之結果儲存，以便將不同完成度之圖片作為DDPM之輸入進行測試。
![output_epoch30](https://hackmd.io/_uploads/r11HPLeLA.png =20%x)![output_epoch250](https://hackmd.io/_uploads/SJlyuUx8A.png =20%x)![output_epoch460](https://hackmd.io/_uploads/H1xsDLgLC.png =20%x)![output_epoch800](https://hackmd.io/_uploads/B1dzO8eLA.png =20%x)![output_epoch2200](https://hackmd.io/_uploads/rk0VuIxLC.png =20%x)
隨著epoch數增加，原先連顏色都不對的圖片，先將顏色的部分導正，隨後圖片的輪廓漸漸出現，直到明顯可以辨識出與原圖相似的特徵。

---

## Denoising Duffusion Probabilistic Model
採用Huggin Face上的預訓練模型，訓練資料集為**celebahq-256**，其預設之DDPMPipeline class中sampling部分為採用random雜訊為原始輸入，修改部分程式碼，達到同時輸入initial image以及epoch的行為。

```python=
# modify here to get input image from the output of DIP
if input_image is None:
    if self.device.type == "mps":
        # randn does not work reproducibly on mps
        image = randn_tensor(image_shape, generator=generator)
        image = image.to(self.device)
    else:
        image = randn_tensor(image_shape, generator=generator, device=self.device)
else:
    image = input_image.to(self.device)
```
![ddpm_generated_image_990 (1)](https://hackmd.io/_uploads/H1oGTXeU0.png =25%x)![ddpm_generated_image_990](https://hackmd.io/_uploads/SkPn4Hl8C.png =25%x)![ddpm_generated_image_990 (1)](https://hackmd.io/_uploads/SJcYrrxUA.png =25%x)![ddpm_generated_image_990 (2)](https://hackmd.io/_uploads/B1Ag8Se8R.png =25%x)


上面示意圖為預設1000次iteration後從隨機噪聲所得之圖片，耗時約 50(s)

---

## Put DIP output as DDPM input
實驗方法為從1開始每10次iteration進行一次輸出並儲存。
首先修改輸出方法，將return改為yield，避免多餘的計算，另外確保為同一輸入中不同t時刻的預測。
```python=
for t in self.progress_bar(self.scheduler.timesteps):
    # 1. predict noise model_output
    model_output = self.unet(image, t).sample
    # 2. compute previous image: x_t -> x_t-1
    image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
    if(t % 10 == 0):
        image_to_output = (image / 2 + 0.5).clamp(0, 1)
        image_to_output = image_to_output.cpu().permute(0, 2, 3, 1).numpy()
        yield ImagePipelineOutput(images=self.numpy_to_pil(image_to_output))
```
![ddpm_generated_image_0](https://hackmd.io/_uploads/ryJGdBxLR.png =20%x)
![ddpm_generated_image_100](https://hackmd.io/_uploads/SJ1G_Be8C.png =20%x)![ddpm_generated_image_200](https://hackmd.io/_uploads/B1Jz_HlUA.png =20%x)![ddpm_generated_image_300](https://hackmd.io/_uploads/SyyMOHe8C.png =20%x)![ddpm_generated_image_400](https://hackmd.io/_uploads/HkyfOHxUC.png =20%x)![ddpm_generated_image_500](https://hackmd.io/_uploads/rk1fOBlL0.png =20%x)
![ddpm_generated_image_600](https://hackmd.io/_uploads/ByyG_reIC.png =20%x)![ddpm_generated_image_700](https://hackmd.io/_uploads/rJJzOHeIR.png =20%x)![ddpm_generated_image_800](https://hackmd.io/_uploads/HkJzdSgIR.png =20%x)![ddpm_generated_image_990](https://hackmd.io/_uploads/HkkzuSeIR.png =20%x)![ddpm_generated_image_900](https://hackmd.io/_uploads/S1kzOBgU0.png =20%x)

首先嘗試了將DIP輸出的圖片輸入，但發現由於time step設定問題，導致unet預測輸出的部分認為此步數下應有相當高的噪聲，也就是圖片的像素分布應該相當隨機，所以將原先的圖片在0~100的輸出期間漸漸地將圖片打亂，最後隨著步數增加才將圖片順利去噪。

**解決方法**
```pythopn
for t in self.progress_bar(self.scheduler.timesteps):
            改為
for t in range(start_step,end_step,-1):
```
指定初始t時刻將會使模型預計的雜訊較少，故可以馬上順利的去除噪聲。

![ddpm_generated_image_0](https://hackmd.io/_uploads/SyP8YHl8C.png =20%x)
![ddpm_generated_image_20](https://hackmd.io/_uploads/Hyv8tSlL0.png =20%x)![ddpm_generated_image_30](https://hackmd.io/_uploads/HyDUtreIC.png =20%x)![ddpm_generated_image_40](https://hackmd.io/_uploads/S1PUKSgIC.png =20%x)![ddpm_generated_image_50](https://hackmd.io/_uploads/rkwUKSeIR.png =20%x)![ddpm_generated_image_60](https://hackmd.io/_uploads/H1vUYHgI0.png =20%x)![ddpm_generated_image_70](https://hackmd.io/_uploads/S1PLFreL0.png =20%x)![ddpm_generated_image_80](https://hackmd.io/_uploads/rJvUFBxLA.png =20%x)![ddpm_generated_image_90](https://hackmd.io/_uploads/ByDUtSgLR.png =20%x)![ddpm_generated_image_100](https://hackmd.io/_uploads/Syw8KSe8R.png =20%x)![ddpm_generated_image_110](https://hackmd.io/_uploads/r1D8FreLA.png =20%x)

初始圖片，採用DIP之輸出，可以發現相同的輸出下從t100->t0補齊了最後的去噪圖像優化過程，最後t0的輸出可見幾乎無噪聲，線條非常柔和。

## 更多測試
將 convolution layer 由 4 改為 6，所得的結果與原先差距不大，會增加模型的複雜度，以及訓練的時長。 
![image](https://hackmd.io/_uploads/rymgEIxIR.png)
![ddpm_generated_image_0 (1)](https://hackmd.io/_uploads/rJZABUeU0.png =20%x)
![ddpm_generated_image_10 (1)](https://hackmd.io/_uploads/BJtV8UeIC.png =20%x)![ddpm_generated_image_20 (1)](https://hackmd.io/_uploads/HyFEI8lLR.png =20%x)![ddpm_generated_image_30 (1)](https://hackmd.io/_uploads/HJF4UUe8C.png =20%x)![ddpm_generated_image_40 (1)](https://hackmd.io/_uploads/S1F4L8gIC.png =20%x)
![ddpm_generated_image_50 (1)](https://hackmd.io/_uploads/SyFNUUg8R.png =20%x)![ddpm_generated_image_60 (1)](https://hackmd.io/_uploads/SktNUIgLA.png =20%x)![ddpm_generated_image_70 (1)](https://hackmd.io/_uploads/H1KVUIgIR.png =20%x)
但在輸出圖像的部分，同樣的epoch下後者更能將顏色表現完整，同時DDPM更可下降到只需50步即有良好的結果。
