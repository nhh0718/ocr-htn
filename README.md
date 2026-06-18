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

Self-hosted OCR API tối ưu cho **tiếng Việt** (có dấu), trả về raw text + blocks (bbox, confidence). Pipeline:

- **Detection:** PaddleOCR (`ch_PP-OCRv4_det`, multilingual)
- **Recognition:** [VietOCR](https://github.com/pbcquoc/vietocr) `vgg_transformer` (SOTA cho tiếng Việt)
- **Extraction:** Pattern-based (regex + anchor, không cần AI) hoặc LLM (Gemini/OpenAI, schema-driven)
- **Serving:** FastAPI + Uvicorn, đóng gói Docker, deploy 1 click lên Render / HF Spaces.

## Endpoints

| Method | Path                       | Mô tả                                                    |
| ------ | -------------------------- | -------------------------------------------------------- |
| GET    | `/`                        | Thông tin service                                        |
| GET    | `/healthz`                 | Health check                                             |
| GET    | `/docs`                    | Swagger UI                                               |
| GET    | `/patterns`                | Liệt kê patterns có sẵn                                  |
| POST   | `/ocr`                     | Multipart upload (`file`) → OCR                          |
| POST   | `/ocr/base64`              | JSON `{ "image_base64": "..." }` → OCR                   |
| POST   | `/extract`                 | Trích xuất trường từ text/ảnh (pattern hoặc LLM)         |
| POST   | `/extract/preset/{name}`   | Upload ảnh + dùng pattern có sẵn                         |

### OCR Response

```json
{
  "text": "Dòng 1\nDòng 2",
  "lines": ["Dòng 1", "Dòng 2"],
  "blocks": [
    {"text": "Dòng", "bbox": [10, 20, 50, 40], "confidence": 1.0, "line_index": 0},
    {"text": "1", "bbox": [55, 20, 70, 40], "confidence": 1.0, "line_index": 0}
  ],
  "image_size": [1024, 768],
  "elapsed_ms": 842
}
```

### Extract Response

```json
{
  "method": "pattern",
  "pattern_id": "acb-transfer",
  "fields": {"amount": 30000, "transaction_id": "5696", "datetime": "15/06/2026 19:06:10"},
  "raw_text": "ACB\nChuyển tiền thành công\n30.000VND\n...",
  "elapsed_ms": 12
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

| Biến                   | Mặc định          | Ý nghĩa                                                         |
| ---------------------- | ----------------- | --------------------------------------------------------------- |
| `VIETOCR_WEIGHTS`      | `vgg_transformer` | `vgg_transformer` (chính xác) hoặc `vgg_seq2seq` (nhẹ hơn)      |
| `OCR_DEVICE`           | `cpu`             | `cpu` hoặc `cuda` (nếu host có GPU)                             |
| `PADDLE_LANG`          | `en`              | Ngôn ngữ detector (multilingual model dùng được cho tiếng Việt) |
| `MAX_IMAGE_SIDE`       | `1600`            | Resize cạnh dài về kích thước này trước khi OCR                 |
| `LINE_Y_TOLERANCE`     | `0.7`             | Hệ số gom box thành dòng                                        |
| `OCR_VERBOSE`          | `false`           | Log thời gian từng request                                      |
| `EXTRACTION_PROVIDER`  | `none`            | `gemini` / `openai` / `none` — bật LLM extraction               |
| `EXTRACTION_API_KEY`   | (empty)           | API key cho provider (Gemini/OpenAI)                            |
| `EXTRACTION_MODEL`     | `gemini-2.0-flash`| Tên model (vd: `gpt-4o-mini`, `gemini-2.0-flash`)               |
| `PATTERNS_DIR`         | `patterns/`       | Thư mục chứa pattern JSON files                                 |

## Tips chất lượng

- Ảnh gốc nên có chiều rộng tối thiểu ~1000px cho text in nhỏ; quá nhỏ recognition sẽ giảm chính xác.
- VietOCR đã train trên tiếng Việt có dấu nên tránh dùng PaddleOCR `recognition` (chất lượng thấp hơn cho VN).
- Với hóa đơn nhiệt mờ: chụp thẳng, đủ sáng; pipeline không tự deskew nặng để giữ tốc độ.

## Cấu trúc project

```
app/
  main.py            FastAPI routes: /ocr, /ocr/base64, /extract, /extract/preset/{name}, /patterns
  ocr_engine.py      Singleton: PaddleOCR (det) + VietOCR (rec) → OCRResult with blocks
  preprocess.py      Decode bytes/base64, EXIF rotate, perspective warp
  extractor.py       Pattern engine: regex + anchor + transform (no AI)
  llm_extractor.py   LLM extraction: Gemini/OpenAI, schema-driven (optional)
  schemas.py         Pydantic request/response models
  config.py          Settings từ env
patterns/
  acb-transfer.json  ACB bank transfer receipt
  momo-receipt.json  MoMo e-wallet receipt
  bidv-transfer.json BIDV bank transfer
  vcb-transfer.json  Vietcombank transfer
  id-card-vn.json    Vietnamese ID card (CCCD)
scripts/
  prefetch_models.py Tải weights ở build time
tests/
  test_api.py        20 tests (stubbed engine, <3s)
Dockerfile
render.yaml
requirements.txt
```

## Extraction — hai chế độ

### 1. Pattern-based (không cần AI)

Pattern là file JSON trong `patterns/`. Mỗi pattern định nghĩa:

- **`detect.any_keyword`**: keywords để auto-detect loại document
- **`fields`**: rules trích xuất từng trường

Mỗi field rule có thể dùng:

| Rule        | Mô tả                                                        |
| ----------- | ----------------------------------------------------------- |
| `regex`     | Regex trên toàn bộ text. Capture group 1 nếu có.            |
| `anchor`    | Tìm dòng chứa keyword, rồi lấy dòng bên dưới / bên phải     |
| `direction` | `below` / `right` / `same_line` (default: `below`)          |
| `stop_at`   | Dừng khi gặp keyword này (cho direction=below)              |
| `ignore`    | Bỏ qua dòng chứa keyword này                                |
| `transform` | `vnd_to_number` / `to_number` / `strip` / `lower` / `upper` |
| `default`   | Giá trị mặc định nếu không extract được                     |

Ví dụ pattern:

```json
{
  "name": "My Receipt",
  "detect": { "any_keyword": ["MY SHOP", "Hóa đơn"] },
  "fields": {
    "total": {
      "regex": "Tổng cộng\\s*([\\d.,]+)\\s*VND",
      "transform": "vnd_to_number"
    },
    "customer": {
      "anchor": "Khách hàng",
      "direction": "below"
    }
  }
}
```

Sử dụng:

```bash
# Auto-detect pattern
curl -X POST http://localhost:7860/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "ACB\nChuyển tiền thành công\n30.000VND\n..."}'

# Chỉ định pattern cụ thể
curl -X POST http://localhost:7860/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "pattern_id": "acb-transfer"}'

# Upload ảnh + pattern
curl -X POST http://localhost:7860/extract/preset/acb-transfer \
  -F "file=@receipt.jpg"
```

### 2. LLM extraction (tùy chọn, cần API key)

Khi không có pattern nào match và request có `schema`, API sẽ gọi LLM để trích xuất:

```bash
# Cấu hình
export EXTRACTION_PROVIDER=gemini  # hoặc openai
export EXTRACTION_API_KEY=your_key
export EXTRACTION_MODEL=gemini-2.0-flash  # hoặc gpt-4o-mini

# Sử dụng
curl -X POST http://localhost:7860/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "OCR text here...",
    "schema": {
      "type": "object",
      "properties": {
        "amount": {"type": "number"},
        "date": {"type": "string"},
        "merchant": {"type": "string"}
      }
    },
    "instructions": "This is a Vietnamese restaurant receipt"
  }'
```

LLM trả về JSON đúng schema — không cần post-processing.

### Luồng xử lý /extract

1. Nếu có `image_base64` → OCR ảnh trước
2. Nếu có `pattern_id` hoặc `pattern` → dùng pattern đó
3. Nếu không → auto-detect pattern từ text
4. Nếu pattern match → extract bằng regex/anchor → trả về `method: "pattern"`
5. Nếu không match + có `schema` → gọi LLM → trả về `method: "llm"`
6. Nếu không match + không có schema → trả về `method: "none"` + raw_text

## License

MIT.
