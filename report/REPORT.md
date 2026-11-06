# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:6/11/2026**

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics: Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh. Vì Image Classification có nhiệm vụ gán nhãn cho toàn bộ khung hình, nhận diện bối cảnh tổng thể thay vì định vị tọa độ từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Đi kèm taxonomy/checkpoint.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Để đảm bảo tính toàn vẹn, tính chính xác, khả năng mở rộng
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Cần phải quy định rõ quy tắc để người gán nhãn chọn một lớp.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì ground truth là thực tế khách quan đã được xác thực còn `score` là xác suất đầu ra của mô hình không phải sự thật tuyệt đối


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái nằm ở vị trí (385.33, 69.24), góc dưới phải nằm ở vị trí (498.92, 348.92), rộng/cao là 113.58/279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.20 có 17 prediction, ngưỡng 0.60 có 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, lượng prediction tăng lên cũng làm gia tăng prediction rác. Reviewer phải kiểm tra kỹ để xóa bỏ các box nhận diện sai.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Luôn ôm sát biên ngoài cùng của phần hiển thị nhìn thấy được của vật thể; nếu vật thể bị che khuất một phần, chỉ vẽ box bao phần lộ diện và tuân theo ranh giới kỹ thuật trong guideline chung.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Ở góc dưới bên trái, bowl bị che khuất bởi bowl khác. Nếu guideline có sẵn quy tắc rõ ràng thì sẽ áp dụng guideline. Nếu có trường hợp không nằm trong guideline thì sẽ đưa lên hỏi Lab Coach (escalation).

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0,70.0],[445.0,71.0],[444.0,71],...]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cung cấp đường viền chi tiết sát theo hình dạng thực tế của vật thể thay vì chỉ là khung chữ nhật thô, giúp phân biệt rõ các chi tiết lõm, lồi hoặc các vật thể đứng gần nhau.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` dùng để phân biệt các object riêng biệt cho dù cùng lớp.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Điểm mút của các đỉnh polygon phải bám sát ranh giới giữa vật thể và nền, không được lẹm vào phần nền quá 2 pixel và phải bao trọn các góc cạnh sắc nét của vật.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Dining table và oven chạm vào nhau. Đối với vùng biên mờ hoặc vật chạm nhau khó tách biệt ranh giới, nếu độ mơ hồ vượt quá quy chuẩn trong guildeline, thực hiện escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn tên lớp đơn hoặc đa nhãn gắn liền với toàn bộ file ảnh. | Ảnh chứa nhiều vật thể gây nhầm lẫn nhãn chính | Gán nhãn phù hợp với taxonomy quy định. | Kiểm tra xem nhãn lớp có phản ánh đúng chủ đề trung tâm của ảnh không. |
| Phát hiện vật thể | Tọa độ hộp (bbox) kèm tên lớp tương ứng. | Box trồng chéo lên nhau ở trong ảnh| Vẽ hộp chính xác bao quanh vật thể theo guideline. | Kiểm tra độ khít của box, lọc bỏ box trùng lặp hoặc thiếu sót. |
| Instance segmentation | Tập hợp điểm tọa độ đa giác (polygon) tạo thành vùng mặt nạ (mask) cho từng cá thể. | Biên vật thể bị lẹm vô vật khác ở trong ảnh |Kẻ chuỗi điểm polygon bám sát biên thực tế của đối tượng.  | Kiểm tra độ tinh xảo của đường viền mask và tính chính xác của instance_id. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Tuyệt đối tuân thủ nguyên tắc bào mật, không sao chép, chia sẻ dữ liệu lab ra khỏi môi trường được cấp phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho người phụ trách trực tiếp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
