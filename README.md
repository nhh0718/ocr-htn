---
title: Vietnamese OCR API
emoji: 📝
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Self-hosted Vietnamese OCR (PaddleOCR + VietOCR)
---

# Vietnamese OCR API

Self-hosted OCR API tối ưu cho **tiếng Việt** (có dấu), trả về raw text. Pipeline:

- **Detection:** PaddleOCR (`ch_PP-OCRv4_det`, multilingual)
- **Recognition:** [VietOCR](https://github.com/pbcquoc/vietocr) `vgg_transformer` (SOTA cho tiếng Việt)
- **Serving:** FastAPI + Uvicorn, đóng gói Docker, deploy 1 click lên Render.

## Endpoints

| Method | Path          | Mô tả                            |
| ------ | ------------- | -------------------------------- |
| GET    | `/`           | Thông tin service                |
| GET    | `/healthz`    | Health check (dùng cho Render)   |
| GET    | `/docs`       | Swagger UI                       |
| POST   | `/ocr`        | Multipart upload (`file`)        |
| POST   | `/ocr/base64` | JSON `{ "image_base64": "..." }` |

Response:

```json
{
  "text": "Dòng 1\nDòng 2",
  "lines": ["Dòng 1", "Dòng 2"],
  "elapsed_ms": 842
}
```

## Chạy local bằng Docker

```bash
docker build -t ocr-vn .
docker run --rm -p 8000:8000 ocr-vn
```

Test:

```bash
curl -F "file=@hoadon.jpg" http://localhost:8000/ocr
```

```bash
# Base64
IMG=$(base64 -w0 hoadon.jpg)
curl -X POST http://localhost:8000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\":\"$IMG\"}"
```

## Chạy local không cần Docker (Python 3.10)

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Lần đầu khởi động sẽ tải weights (~500MB–1GB) về cache mặc định của VietOCR/Paddle.

## Deploy lên Hugging Face Spaces (miễn phí, khuyến nghị)

HF Spaces tier miễn phí cho **16GB RAM** + **2 vCPU** — dư sức chạy VietOCR `vgg_transformer`.

1. Tạo Space mới: https://huggingface.co/new-space → **SDK: Docker** → blank template.
2. Copy URL git của Space (vd: `https://huggingface.co/spaces/<user>/ocr-htn`).
3. Push code:

```bash
git remote add hf https://huggingface.co/spaces/<user>/ocr-htn
git push hf main
```

> Lần đầu HF sẽ hỏi token (Settings → Access Tokens → write). Build mất ~10–15 phút.

Endpoint: `https://<user>-ocr-htn.hf.space/ocr`. Frontmatter trong `README.md` (sdk, app_port, ...) là cấu hình bắt buộc của HF Spaces.

## Deploy lên Render

1. Push repo này lên GitHub.
2. Trên Render Dashboard → **New** → **Blueprint** → trỏ đến repo. Render sẽ đọc `render.yaml`.
3. Chọn plan **Standard** trở lên (cần ≥ 2GB RAM cho VietOCR `vgg_transformer`). Free plan 512MB sẽ OOM.
4. Deploy. Endpoint sẽ là `https://<service>.onrender.com`.

> **Lưu ý:** Cold start lần đầu (sau idle / redeploy) khoảng 15–25s do load model. Sau đó các request chỉ tốn 0.3–1.5s/ảnh tùy kích thước.

## Cấu hình qua biến môi trường

| Biến               | Mặc định          | Ý nghĩa                                                         |
| ------------------ | ----------------- | --------------------------------------------------------------- |
| `VIETOCR_WEIGHTS`  | `vgg_transformer` | `vgg_transformer` (chính xác) hoặc `vgg_seq2seq` (nhẹ hơn)      |
| `OCR_DEVICE`       | `cpu`             | `cpu` hoặc `cuda` (nếu host có GPU)                             |
| `PADDLE_LANG`      | `en`              | Ngôn ngữ detector (multilingual model dùng được cho tiếng Việt) |
| `MAX_IMAGE_SIDE`   | `1600`            | Resize cạnh dài về kích thước này trước khi OCR                 |
| `LINE_Y_TOLERANCE` | `0.7`             | Hệ số gom box thành dòng                                        |
| `OCR_VERBOSE`      | `false`           | Log thời gian từng request                                      |

## Tips chất lượng

- Ảnh gốc nên có chiều rộng tối thiểu ~1000px cho text in nhỏ; quá nhỏ recognition sẽ giảm chính xác.
- VietOCR đã train trên tiếng Việt có dấu nên tránh dùng PaddleOCR `recognition` (chất lượng thấp hơn cho VN).
- Với hóa đơn nhiệt mờ: chụp thẳng, đủ sáng; pipeline không tự deskew nặng để giữ tốc độ.

## Cấu trúc project

```
app/
  main.py           FastAPI routes + lifespan warmup
  ocr_engine.py     Singleton: PaddleOCR (det) + VietOCR (rec) + line grouping
  preprocess.py     Decode bytes/base64, EXIF rotate, perspective warp
  schemas.py        Pydantic request/response
  config.py         Settings từ env
scripts/
  prefetch_models.py  Tải weights ở build time
tests/
  test_ocr.py
Dockerfile
render.yaml
requirements.txt
```

## License

MIT.
