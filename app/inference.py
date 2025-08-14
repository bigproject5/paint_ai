import time
import requests
import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
 
# 파일의 절대 경로를 얻기 위한 os 모듈 import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
 
from schemas import *
from config import config
 
class PaintDefectDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()
   
    def _load_model(self):
        try:
            if not YOLO_AVAILABLE:
                return
           
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_abs_path = os.path.join(base_dir, '../models/best.onnx')
 
            model_paths = [
                model_abs_path,
                config.MODEL_PATH,
                "../models/best.onnx",
                "../models/best.onnx",
                "./models/best.pt",
                "best.pt"
            ]
           
            for model_path in model_paths:
                if not model_path:
                    continue
                try:
                    if os.path.exists(model_path):
                        self.model = YOLO(model_path)
                        self.model_loaded = True
                        return
                    else:
                        continue
                except Exception:
                    continue
               
        except Exception:
            pass
   
    def download_image(self, image_path: str) -> np.ndarray:
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, timeout=config.DOWNLOAD_TIMEOUT)
                response.raise_for_status()
               
                if len(response.content) > config.MAX_IMAGE_SIZE:
                    raise Exception(f"이미지 크기가 너무 큽니다")
               
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                if not os.path.exists(image_path):
                    raise Exception(f"이미지 파일을 찾을 수 없습니다: {image_path}")
               
                image = cv2.imread(image_path)
           
            if image is None:
                raise Exception("이미지를 로드할 수 없습니다")
               
            return image
           
        except Exception as e:
            raise Exception(f"이미지 로드 실패: {e}")
   
    def detect_defects(self, collect_data_path: str) -> Dict:
        if not self.model_loaded:
            raise Exception("YOLO 모델이 로드되지 않았습니다")
       
        start_time = time.time()
       
        try:
            image = self.download_image(collect_data_path)
           
            results = self.model.predict(
                source=image,
                conf=config.CONFIDENCE_THRESHOLD,
                save=False,
                verbose=False
            )
           
            defects = []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
               
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                   
                    class_name = self.model.names.get(class_id, f"unknown_class_{class_id}")
                    defect_type = self._map_class_to_defect_type(class_name)
                   
                    area = (x2 - x1) * (y2 - y1)
                    severity = self._calculate_severity(area, confidence, defect_type)
                   
                    defect = DetectedDefect(
                        defect_type=defect_type,
                        confidence=confidence,
                        bbox={
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(x2 - x1),
                            "height": float(y2 - y1)
                        },
                        severity=severity
                    )
                    defects.append(defect)
           
            processing_time = time.time() - start_time
            quality_score = self._calculate_quality_score(defects)
            overall_grade = self._determine_quality_grade(defects, quality_score)
            is_defect = len(defects) > 0
           
            return {
                "defects": defects,
                "quality_score": quality_score,
                "overall_grade": overall_grade,
                "processing_time": processing_time,
                "is_defect": is_defect
            }
           
        except Exception as e:
            raise Exception(f"결함 검출 실패: {e}")
   
    def _map_class_to_defect_type(self, class_name: str) -> DefectType:
        class_lower = class_name.lower()
       
        if any(keyword in class_lower for keyword in ["dent", "덴트", "찌그러짐", "pdr"]):
            return DefectType.PDR_DENT
        elif any(keyword in class_lower for keyword in [
            "paint", "페인트", "도장", "색상", "color",
            "bubble", "기포", "crack", "균열", "orange",
            "run", "흘러내림", "spray", "스프레이"
        ]):
            return DefectType.PAINT
        else:
            return DefectType.SCRATCH
   
    def _calculate_severity(self, area: float, confidence: float, defect_type: DefectType) -> float:
        base_severity = confidence
        area_factor = min(area / 10000, 1.0)
       
        type_weights = {
            DefectType.SCRATCH: 1.0,
            DefectType.PDR_DENT: 1.3,
            DefectType.PAINT: 1.1
        }
       
        type_weight = type_weights.get(defect_type, 1.0)
        severity = min((base_severity + area_factor * 0.3) * type_weight, 1.0)
       
        return severity
   
    def _calculate_quality_score(self, defects: List[DetectedDefect]) -> float:
        if not defects:
            return 1.0
       
        total_deduction = 0
       
        for defect in defects:
            base_deduction = defect.severity * defect.confidence * 0.1
           
            if defect.defect_type == DefectType.PDR_DENT:
                base_deduction *= 1.2
            elif defect.defect_type == DefectType.PAINT:
                base_deduction *= 1.1
               
            total_deduction += base_deduction
       
        quality_score = max(1.0 - total_deduction, 0.0)
        return quality_score
   
    def _determine_quality_grade(self, defects: List[DetectedDefect], quality_score: float) -> QualityGrade:
        critical_defects = sum(1 for d in defects if d.severity > config.CRITICAL_SEVERITY_THRESHOLD)
        has_pdr_dent = any(d.defect_type == DefectType.PDR_DENT for d in defects)
       
        if critical_defects > 2:
            return QualityGrade.REJECT
        elif has_pdr_dent and critical_defects > 0:
            return QualityGrade.REJECT
        elif quality_score < config.QUALITY_MINOR_THRESHOLD:
            return QualityGrade.MAJOR_DEFECT
        elif quality_score < config.QUALITY_PASS_THRESHOLD:
            return QualityGrade.MINOR_DEFECT
        elif has_pdr_dent:
            return QualityGrade.MINOR_DEFECT
        else:
            return QualityGrade.PASS
 
detector = PaintDefectDetector()
 
def process_ai_diagnosis(event_data: TestStartedEventDTO) -> AiDiagnosisCompletedEventDTO:
    try:
        result = detector.detect_defects(event_data.collect_data_path)
       
        diagnosis_result = {
            "overall_grade": result["overall_grade"].value,
            "quality_score": result["quality_score"],
            "defects_found": [
                {
                    "defect_type": defect.defect_type.value,
                    "confidence": defect.confidence,
                    "bbox": defect.bbox,
                    "severity": defect.severity
                }
                for defect in result["defects"]
            ],
            "total_defects": len(result["defects"]),
            "processing_time": result["processing_time"],
            "inspection_date": datetime.now().isoformat()
        }
       
        result_data_path = generate_result_path(event_data.collect_data_path, event_data.inspection_id)
        save_diagnosis_result(diagnosis_result, result_data_path)
       
        completed_event = AiDiagnosisCompletedEventDTO(
            auditId=event_data.audit_id,
            inspection_id=event_data.inspection_id,
            inspection_type=event_data.inspection_type,
            is_defect=result["is_defect"],
            collect_data_path=event_data.collect_data_path,
            result_data_path=result_data_path,
            diagnosis_result=json.dumps(diagnosis_result, ensure_ascii=False)
        )
       
        return completed_event
       
    except Exception as e:
        error_result = {
            "error": str(e),
            "overall_grade": "error",
            "quality_score": 0.0,
            "defects_found": [],
            "total_defects": 0,
            "processing_time": 0.0,
            "inspection_date": datetime.now().isoformat()
        }
       
        result_data_path = generate_result_path(event_data.collect_data_path, event_data.inspection_id, error=True)
        save_diagnosis_result(error_result, result_data_path)
       
        return AiDiagnosisCompletedEventDTO(
            audit_id=event_data.audit_id,
            inspection_id=event_data.inspection_id,
            inspection_type=event_data.inspection_type,
            is_defect=False,
            collect_data_path=event_data.collect_data_path,
            result_data_path=result_data_path,
            diagnosis_result=json.dumps(error_result, ensure_ascii=False)
        )
 
def generate_result_path(collect_data_path: str, inspection_id: int, error: bool = False) -> str:
    try:
        directory = os.path.dirname(collect_data_path)
        filename = os.path.basename(collect_data_path)
        name, ext = os.path.splitext(filename)
       
        if error:
            result_filename = f"{name}_result_error_{inspection_id}.json"
        else:
            result_filename = f"{name}_result_{inspection_id}.json"
       
        result_directory = os.path.join(directory, "results")
        os.makedirs(result_directory, exist_ok=True)
       
        return os.path.join(result_directory, result_filename)
       
    except Exception:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"./results/diagnosis_result_{inspection_id}_{timestamp}.json"
 
def save_diagnosis_result(result_data: dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
       
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
           
    except Exception as e:
        raise
 
def get_model_status() -> dict:
    model_path_str = str(config.MODEL_PATH) if config.MODEL_PATH else ""
 
    return {
        "model_loaded": detector.model_loaded,
        "yolo_available": YOLO_AVAILABLE,
        "model_path": model_path_str,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "available": detector.model_loaded
    }
 
def process_paint_inspection(request: PaintInspectionRequest) -> PaintInspectionResponse:
    try:
        result = detector.detect_defects(request.image_url)
       
        response = PaintInspectionResponse(
            car_id=request.car_id,
            part_code=request.part_code,
            overall_grade=result["overall_grade"].value,
            quality_score=result["quality_score"],
            defects_found=result["defects"],
            total_defects=len(result["defects"]),
            processing_time=result["processing_time"],
            inspection_date=datetime.now()
        )
       
        return response
       
    except Exception as e:
        raise Exception(f"검사 처리 실패: {e}")