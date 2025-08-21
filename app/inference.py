import time
import requests
import cv2
import numpy as np
import json
import os
import boto3
from datetime import datetime
from typing import List, Dict, Optional, Union
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from schemas import *
from config import config

class S3Uploader:
    def __init__(self):
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
    
    # 수정: 비디오 처리를 위한 새로운 메서드 추가
    def upload_defect_video(self, frames_with_defects: List[np.ndarray], inspection_id: int) -> Optional[str]:
        """비디오 프레임들을 mp4로 저장하여 S3에 업로드"""
        if not IMAGEIO_AVAILABLE:
            print("imageio 라이브러리가 필요합니다")
            return None
            
        try:
            if not self.bucket_name:
                return None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_video_path = f"temp_defect_video_{inspection_id}_{timestamp}.mp4"
            
            # 비디오 생성
            with imageio.get_writer(temp_video_path, fps=10) as writer:
                for frame in frames_with_defects:
                    # BGR을 RGB로 변환
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb_frame)
            
            # S3 업로드
            with open(temp_video_path, 'rb') as f:
                video_data = f.read()
            
            s3_key = f"defect_videos/{timestamp}/inspection_{inspection_id}_defects.mp4"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=video_data,
                ContentType='video/mp4'
            )
            
            # 임시 파일 삭제
            os.remove(temp_video_path)
            return f"s3://{self.bucket_name}/{s3_key}"
            
        except Exception as e:
            print(f"비디오 업로드 실패: {e}")
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return None
   
    def upload_defect_media(self, media: np.ndarray, inspection_id: int, defects: List[DetectedDefect], is_video: bool = False) -> Optional[str]:
        try:
            if not self.bucket_name:
                return None
                
            annotated_media = self._draw_defect_boxes(media.copy(), defects)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 수정: 비디오는 단일 프레임만 이미지로 저장
            _, buffer = cv2.imencode('.jpg', annotated_media, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if is_video:
                s3_key = f"defect_images/{timestamp}/inspection_{inspection_id}_video_sample.jpg"
            else:
                s3_key = f"defect_images/{timestamp}/inspection_{inspection_id}_defects.jpg"
            content_type = 'image/jpeg'
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.tobytes(),
                ContentType=content_type
            )
            
            return f"s3://{self.bucket_name}/{s3_key}"
            
        except Exception:
            return None
   
    def _draw_defect_boxes(self, media: np.ndarray, defects: List[DetectedDefect]) -> np.ndarray:
        colors = {
            DefectType.SCRATCH: (0, 255, 0),
            DefectType.PDR_DENT: (0, 0, 255),
            DefectType.PAINT: (255, 0, 0)
        }
        
        for defect in defects:
            bbox = defect.bbox
            x1, y1 = int(bbox["x"]), int(bbox["y"])
            x2, y2 = x1 + int(bbox["width"]), y1 + int(bbox["height"])
            
            color = colors.get(defect.defect_type, (255, 255, 255))
            
            cv2.rectangle(media, (x1, y1), (x2, y2), color, 2)
            
            label = f"{defect.defect_type.value}: {defect.confidence:.2f}"
            cv2.putText(media, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return media

class PaintDefectDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.s3_uploader = S3Uploader()
        self._load_model()
   
    def _load_model(self):
        try:
            if not YOLO_AVAILABLE:
                return
           
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, '../models/car_damage_model.onnx')
           
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_loaded = True
        except Exception:
            pass
   
    def download_media(self, media_path: str) -> Union[np.ndarray, cv2.VideoCapture]:
        try:
            if media_path.startswith(('http://', 'https://')):
                response = requests.get(media_path, timeout=config.DOWNLOAD_TIMEOUT)
                response.raise_for_status()
               
                if len(response.content) > config.MAX_IMAGE_SIZE:
                    raise Exception("미디어 크기가 너무 큽니다")
               
                if self._is_video_url(media_path):
                    temp_path = f"temp_video_{int(time.time())}.mp4"
                    with open(temp_path, 'wb') as f:
                        f.write(response.content)
                    return cv2.VideoCapture(temp_path)
                else:
                    image_array = np.frombuffer(response.content, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    return image
            else:
                if not os.path.exists(media_path):
                    raise Exception(f"미디어 파일을 찾을 수 없습니다: {media_path}")
               
                if self._is_video_file(media_path):
                    return cv2.VideoCapture(media_path)
                else:
                    image = cv2.imread(media_path)
                    return image
           
        except Exception as e:
            raise Exception(f"미디어 로드 실패: {e}")
    
    def _is_video_url(self, url: str) -> bool:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        return any(url.lower().endswith(ext) for ext in video_extensions)
    
    def _is_video_file(self, path: str) -> bool:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        return any(path.lower().endswith(ext) for ext in video_extensions)
   
    def detect_defects(self, collect_data_path: str, inspection_id: int) -> Dict:
        if not self.model_loaded:
            raise Exception("YOLO 모델이 로드되지 않았습니다")
       
        start_time = time.time()
        
        try:
            media = self.download_media(collect_data_path)
            is_video = isinstance(media, cv2.VideoCapture)
            
            if is_video:
                return self._process_video(media, inspection_id, start_time)
            else:
                return self._process_image(media, inspection_id, start_time)
                
        except Exception as e:
            raise Exception(f"결함 검출 실패: {e}")
    
    def _process_image(self, image: np.ndarray, inspection_id: int, start_time: float) -> Dict:
        if image is None:
            raise Exception("이미지를 로드할 수 없습니다")
            
        results = self.model.predict(
            source=image,
            conf=config.CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False
        )
        
        defects = self._extract_defects(results)
        processing_time = time.time() - start_time
        
        return self._build_result(defects, processing_time, inspection_id, image, False, image.shape)
    
    # 수정: 비디오 처리에서 결함 있는 프레임들 수집
    def _process_video(self, video: cv2.VideoCapture, inspection_id: int, start_time: float) -> Dict:
        all_defects = []
        frame_count = 0
        analyzed_frames = 0
        sample_frame = None
        defect_frames = []
        
        # 비디오 정보 가져오기
        fps = video.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            frame_count += 1
            if sample_frame is None:
                sample_frame = frame.copy()
            
            if frame_count % 5 != 0:
                continue
                
            analyzed_frames += 1
            results = self.model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                save=False,
                verbose=False
            )
            
            frame_defects = self._extract_defects(results)
            if frame_defects:
                all_defects.extend(frame_defects)
                annotated_frame = self.s3_uploader._draw_defect_boxes(frame.copy(), frame_defects)
                defect_frames.append(annotated_frame)
        
        video.release()
        
        unique_defects = self._merge_similar_defects(all_defects)
        processing_time = time.time() - start_time
        quality_score = self._calculate_quality_score(unique_defects)
        overall_grade = self._determine_quality_grade(unique_defects, quality_score)
        
        # 비디오 업로드 (이미지가 아닌 영상으로)
        defect_media_path = None
        if defect_frames and IMAGEIO_AVAILABLE:
            defect_media_path = self.s3_uploader.upload_defect_video(defect_frames, inspection_id)
        elif sample_frame is not None:
            defect_media_path = self.s3_uploader.upload_defect_media(
                sample_frame, inspection_id, unique_defects, True
            )
        
        # 비디오 정보 포함
        frame_info = {
            "duration": duration,
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames
        }
        
        defect_message = self._generate_defect_message(
            unique_defects, 
            sample_frame.shape if sample_frame is not None else (480, 640, 3),
            processing_time,
            quality_score,
            overall_grade.value,
            is_video=True,
            frame_info=frame_info
        )
        
        return {
            "defects": unique_defects,
            "quality_score": quality_score,
            "overall_grade": overall_grade,
            "processing_time": processing_time,
            "is_defect": len(unique_defects) > 0,
            "defect_image_path": defect_media_path,
            "defect_message": defect_message
        }
    
    def _extract_defects(self, results) -> List[DetectedDefect]:
        defects = []
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
           
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                except:
                    x1, y1, x2, y2 = box.xyxy[0].numpy() if hasattr(box.xyxy[0], 'numpy') else box.xyxy[0]
                    confidence = float(box.conf[0].numpy() if hasattr(box.conf[0], 'numpy') else box.conf[0])
                    class_id = int(box.cls[0].numpy() if hasattr(box.cls[0], 'numpy') else box.cls[0])
               
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
        
        return defects
    
    def _merge_similar_defects(self, defects: List[DetectedDefect]) -> List[DetectedDefect]:
        if not defects:
            return []
        
        merged = []
        threshold = 50
        
        for defect in defects:
            is_similar = False
            
            for merged_defect in merged:
                if (defect.defect_type == merged_defect.defect_type and
                    abs(defect.bbox["x"] - merged_defect.bbox["x"]) < threshold and
                    abs(defect.bbox["y"] - merged_defect.bbox["y"]) < threshold):
                    
                    if defect.confidence > merged_defect.confidence:
                        merged_defect.confidence = defect.confidence
                        merged_defect.severity = defect.severity
                    
                    is_similar = True
                    break
            
            if not is_similar:
                merged.append(defect)
        
        return merged
    
    
    # 새로운 추가: 완전한 문장형 결과 메시지 생성
    def _generate_defect_message(self, defects: List[DetectedDefect], image_shape, processing_time: float, 
                                quality_score: float, overall_grade: str, is_video: bool = False, 
                                frame_info: dict = None) -> str:
        """분석 결과를 완전한 문장으로 변환"""
        
        type_names = {"scratch": "스크래치", "pdr_dent": "덴트", "paint": "도장"}
        
        # 평균 신뢰도 계산
        avg_confidence = sum(d.confidence for d in defects) / len(defects) if defects else 1.0
        confidence_percent = avg_confidence * 100
        
        # 미디어 타입별 분석 정보
        if is_video and frame_info:
            media_info = f"영상분석 {frame_info.get('duration', 0):.1f}초 {frame_info.get('analyzed_frames', 0)}프레임"
        else:
            media_info = "이미지분석 1장"
        
        if not defects:
            return (f"{media_info}에 대해 차량 외관을 종합 검사한 결과, 결함이 발견되지 않았습니다. "
                   f"품질점수 {quality_score:.2f}, 신뢰도 {confidence_percent:.1f}%, 처리시간 {processing_time:.1f}초로 "
                   f"양호한 상태로 판정되었습니다.")
        
        # 결함 정보 정리
        type_counts = {}
        for defect in defects:
            type_name = type_names.get(defect.defect_type.value, defect.defect_type.value)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # 결함 설명 생성
        defect_descriptions = []
        for type_name, count in type_counts.items():
            if count == 1:
                defect_descriptions.append(f"{type_name} 결함")
            else:
                defect_descriptions.append(f"{type_name} 결함 {count}개")
        
        defect_text = ", ".join(defect_descriptions)
        
        # 조치 필요성 판단
        if overall_grade in ["major_defect", "reject"]:
            action_text = "즉시 수리 필요한"
        elif overall_grade == "minor_defect":
            action_text = "점검 필요한"
        else:
            action_text = "양호한"
        
        return (f"{media_info}에 대해 차량 외관을 검사한 결과, {defect_text}이 발견되었습니다. "
               f"품질점수 {quality_score:.2f}, 신뢰도 {confidence_percent:.1f}%, 처리시간 {processing_time:.1f}초로 "
               f"{action_text} 상태로 판정되었습니다.")
    
    # 수정: 이미지 형태 정보 추가
    def _build_result(self, defects: List[DetectedDefect], processing_time: float, 
                     inspection_id: int, sample_media: np.ndarray, is_video: bool, image_shape) -> Dict:
        quality_score = self._calculate_quality_score(defects)
        overall_grade = self._determine_quality_grade(defects, quality_score)
        is_defect = len(defects) > 0
        
        defect_media_path = None
        if is_defect:
            defect_media_path = self.s3_uploader.upload_defect_media(
                sample_media, inspection_id, defects, is_video
            )
        
        # 수정: 메시지 생성에 모든 필요 정보 전달
        defect_message = self._generate_defect_message(
            defects, 
            image_shape,
            processing_time,
            quality_score,
            overall_grade.value,
            is_video=is_video
        )
       
        return {
            "defects": defects,
            "quality_score": quality_score,
            "overall_grade": overall_grade,
            "processing_time": processing_time,
            "is_defect": is_defect,
            "defect_image_path": defect_media_path,
            "defect_message": defect_message
        }
   
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
  
        if len(defects) == 0:
            return QualityGrade.PASS
        elif critical_defects > 2:
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
            return QualityGrade.MINOR_DEFECT

detector = PaintDefectDetector()

# 수정: 결과에 문장형 메시지 포함
def process_ai_diagnosis(event_data: TestStartedEventDTO) -> AiDiagnosisCompletedEventDTO:
    try:
        result = detector.detect_defects(event_data.collect_data_path, event_data.inspection_id)
       
        defect_types = list(set([defect.defect_type.value for defect in result["defects"]]))
       
        diagnosis_result = SimpleDiagnosisResult(
            grade=result["overall_grade"].value,
            score=round(result["quality_score"], 3),
            defect_count=len(result["defects"]),
            defect_types=defect_types,
            processing_time=round(result["processing_time"], 3)
        )
       
        detailed_result = DiagnosisResultDetail(
            overall_grade=result["overall_grade"].value,
            quality_score=result["quality_score"],
            defects_found=[
                {
                    "defect_type": defect.defect_type.value,
                    "confidence": defect.confidence,
                    "bbox": defect.bbox,
                    "severity": defect.severity
                }
                for defect in result["defects"]
            ],
            total_defects=len(result["defects"]),
            processing_time=result["processing_time"],
            inspection_date=datetime.now().isoformat()
        )
       
        if result["is_defect"] and result.get("defect_image_path"):
            result_data_path = result["defect_image_path"]
        else:
            result_data_path = None
       
        # 수정: JSON 최소화 + 문장형 메시지는 자세하게
        completed_event = AiDiagnosisCompletedEventDTO(
            audit_id=event_data.audit_id,
            inspection_id=event_data.inspection_id,
            inspection_type=event_data.inspection_type,
            is_defect=result["is_defect"],
            collect_data_path=event_data.collect_data_path,
            result_data_path=result_data_path,
            diagnosis_result=result.get("defect_message", "")  # JSON 대신 문장형 메시지만 저장
        )
       
        return completed_event
       
    except Exception as e:
        error_result = SimpleDiagnosisResult(
            grade="error",
            score=0.0,
            defect_count=0,
            defect_types=[],
            processing_time=0.0,
            error=str(e)
        )
       
        return AiDiagnosisCompletedEventDTO(
            audit_id=event_data.audit_id,
            inspection_id=event_data.inspection_id,
            inspection_type=event_data.inspection_type,
            is_defect=False,
            collect_data_path=event_data.collect_data_path,
            result_data_path=None,
            diagnosis_result=str(e)  # 에러 시에도 문자열만 저장
        )

def get_model_status() -> dict:
    model_path_str = str(config.MODEL_PATH) if config.MODEL_PATH else ""

    return {
        "model_loaded": detector.model_loaded,
        "yolo_available": YOLO_AVAILABLE,
        "imageio_available": IMAGEIO_AVAILABLE,  # 수정: imageio 상태 추가
        "model_path": model_path_str,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "available": detector.model_loaded
    }

def process_paint_inspection(request: PaintInspectionRequest) -> PaintInspectionResponse:
    try:
        temp_inspection_id = int(time.time())
        result = detector.detect_defects(request.image_url, temp_inspection_id)
       
        response = PaintInspectionResponse(
            car_id=request.car_id,
            part_code=request.part_code,
            overall_grade=result["overall_grade"],
            quality_score=result["quality_score"],
            defects_found=result["defects"],
            total_defects=len(result["defects"]),
            processing_time=result["processing_time"],
            inspection_date=datetime.now()
        )
       
        return response
       
    except Exception as e:
        raise Exception(f"검사 처리 실패: {e}")