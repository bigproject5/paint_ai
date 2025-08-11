import time
import requests
import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics가 설치되지 않았습니다. pip install ultralytics")

from app.schemas import *
from app.config import config

class PaintDefectDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """YOLO 모델 로드"""
        try:
            if not YOLO_AVAILABLE:
                print("❌ YOLO를 사용할 수 없습니다")
                return
            
            # 모델 경로 우선순위
            model_paths = [
                config.MODEL_PATH,
                "runs_yolo11/car_defect_v2/weights/best.pt",
                "./models/best.pt",
                "best.pt"
            ]
            
            for model_path in model_paths:
                try:
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    print(f"✅ YOLO 모델 로드 완료: {model_path}")
                    return
                except Exception as e:
                    if config.DEBUG:
                        print(f"⚠️ {model_path} 로드 실패: {e}")
                    continue
            
            print("❌ 사용 가능한 모델 파일을 찾을 수 없습니다")
                
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
    
    def download_image(self, image_path: str) -> np.ndarray:
        """로컬 파일 또는 URL에서 이미지 로드"""
        try:
            # URL인지 로컬 파일인지 확인
            if image_path.startswith(('http://', 'https://')):
                # URL에서 이미지 다운로드
                response = requests.get(image_path, timeout=config.DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                
                # 파일 크기 확인
                if len(response.content) > config.MAX_IMAGE_SIZE:
                    raise Exception(f"이미지 크기가 너무 큽니다. 최대 {config.MAX_IMAGE_SIZE // (1024*1024)}MB")
                
                # 이미지 디코딩
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                # 로컬 파일에서 이미지 로드
                if not os.path.exists(image_path):
                    raise Exception(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                
                image = cv2.imread(image_path)
            
            if image is None:
                raise Exception("이미지를 로드할 수 없습니다")
                
            return image
            
        except Exception as e:
            raise Exception(f"이미지 로드 실패: {e}")
    
    def detect_defects(self, collect_data_path: str) -> Dict:
        """도장면 결함 검출"""
        if not self.model_loaded:
            raise Exception("YOLO 모델이 로드되지 않았습니다")
        
        start_time = time.time()
        
        try:
            # 이미지 로드
            image = self.download_image(collect_data_path)
            
            # YOLO 추론 실행
            results = self.model.predict(
                source=image,
                conf=config.CONFIDENCE_THRESHOLD,
                save=False,
                verbose=False
            )
            
            # 결과 파싱
            defects = []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    # YOLO가 제공한 바운딩박스 좌표 가져오기
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 클래스명 매핑
                    class_name = self.model.names.get(class_id, f"unknown_class_{class_id}")
                    defect_type = self._map_class_to_defect_type(class_name)
                    
                    # 결함 면적 계산
                    area = (x2 - x1) * (y2 - y1)
                    
                    # 심각도 계산
                    severity = self._calculate_severity(area, confidence, defect_type)
                    
                    # DetectedDefect 객체 생성
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
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 전체 품질 점수 및 등급 계산
            quality_score = self._calculate_quality_score(defects)
            overall_grade = self._determine_quality_grade(defects, quality_score)
            
            # 결함 여부 판단 (결함이 있으면 True)
            is_defect = len(defects) > 0
            
            if config.DEBUG:
                print(f"🔍 검출 완료: {len(defects)}개 결함 발견")
                for defect in defects:
                    print(f"  - {defect.defect_type.value}: {defect.confidence:.2f}")
            
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
        """YOLO 클래스를 결함 유형으로 매핑"""
        # 클래스명을 소문자로 변환해서 매핑
        class_lower = class_name.lower()
        
        # PDR_DENT 관련 키워드
        if any(keyword in class_lower for keyword in ["dent", "덴트", "찌그러짐", "pdr"]):
            return DefectType.PDR_DENT
        
        # PAINT 관련 키워드 (도장 결함들)
        elif any(keyword in class_lower for keyword in [
            "paint", "페인트", "도장", "색상", "color", 
            "bubble", "기포", "crack", "균열", "orange", 
            "run", "흘러내림", "spray", "스프레이"
        ]):
            return DefectType.PAINT
        
        # SCRATCH 관련 키워드 (기본값으로도 사용)
        else:
            return DefectType.SCRATCH
    
    def _calculate_severity(self, area: float, confidence: float, defect_type: DefectType) -> float:
        """결함 심각도 계산"""
        # 기본 심각도 = 신뢰도
        base_severity = confidence
        
        # 면적 요소 (면적이 클수록 심각)
        area_factor = min(area / 10000, 1.0)  # 10000px²을 기준으로 정규화
        
        # 결함 유형별 가중치
        type_weights = {
            DefectType.SCRATCH: 1.0,      # 스크래치 - 기본
            DefectType.PDR_DENT: 1.3,     # PDR 덴트 - 더 심각
            DefectType.PAINT: 1.1         # 페인트 결함 - 약간 심각
        }
        
        type_weight = type_weights.get(defect_type, 1.0)
        
        # 최종 심각도 = (기본 심각도 + 면적 요소) × 유형 가중치
        severity = min((base_severity + area_factor * 0.3) * type_weight, 1.0)
        
        return severity
    
    def _calculate_quality_score(self, defects: List[DetectedDefect]) -> float:
        """전체 품질 점수 계산 (1.0 = 완벽, 0.0 = 최악)"""
        if not defects:
            return 1.0  # 결함이 없으면 완벽한 점수
        
        total_deduction = 0
        
        for defect in defects:
            # 기본 차감 = 심각도 × 신뢰도 × 0.1
            base_deduction = defect.severity * defect.confidence * 0.1
            
            # 결함 유형별 추가 차감
            if defect.defect_type == DefectType.PDR_DENT:
                base_deduction *= 1.2  # PDR 덴트는 20% 더 차감
            elif defect.defect_type == DefectType.PAINT:
                base_deduction *= 1.1  # 페인트 결함은 10% 더 차감
                
            total_deduction += base_deduction
        
        # 최종 점수 (최소 0.0)
        quality_score = max(1.0 - total_deduction, 0.0)
        
        return quality_score
    
    def _determine_quality_grade(self, defects: List[DetectedDefect], quality_score: float) -> QualityGrade:
        """품질 등급 결정"""
        # 심각한 결함 개수 (심각도 > 0.7)
        critical_defects = sum(1 for d in defects if d.severity > config.CRITICAL_SEVERITY_THRESHOLD)
        
        # PDR_DENT 존재 여부
        has_pdr_dent = any(d.defect_type == DefectType.PDR_DENT for d in defects)
        
        # 등급 결정 로직
        if critical_defects > 2:
            return QualityGrade.REJECT  # 심각한 결함 3개 이상 → 불합격
        elif has_pdr_dent and critical_defects > 0:
            return QualityGrade.REJECT  # PDR 덴트 + 심각한 결함 → 불합격
        elif quality_score < config.QUALITY_MINOR_THRESHOLD:  # < 0.6
            return QualityGrade.MAJOR_DEFECT
        elif quality_score < config.QUALITY_PASS_THRESHOLD:   # < 0.8
            return QualityGrade.MINOR_DEFECT
        elif has_pdr_dent:
            return QualityGrade.MINOR_DEFECT  # PDR 덴트가 있으면 최소 경미한 결함
        else:
            return QualityGrade.PASS

# 전역 인스턴스
detector = PaintDefectDetector()

def process_ai_diagnosis(event_data: TestStartedEventDTO) -> AiDiagnosisCompletedEventDTO:
    """AI 진단 처리 (TestStartedEvent를 받아 처리)"""
    try:
        # YOLO 모델로 결함 검출
        result = detector.detect_defects(event_data.collect_data_path)
        
        # 진단 결과 JSON 생성
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
        
        # 결과 파일 저장 경로 생성
        result_data_path = generate_result_path(event_data.collect_data_path, event_data.inspection_id)
        
        # 결과를 파일로 저장
        save_diagnosis_result(diagnosis_result, result_data_path)
        
        # AiDiagnosisCompletedEventDTO 생성
        completed_event = AiDiagnosisCompletedEventDTO(
            audit_id=event_data.audit_id,
            inspection_id=event_data.inspection_id,
            inspection_type=event_data.inspection_type,
            is_defect=result["is_defect"],
            collect_data_path=event_data.collect_data_path,
            result_data_path=result_data_path,
            diagnosis_result=json.dumps(diagnosis_result, ensure_ascii=False)
        )
        
        # 디버그 정보 출력
        if config.DEBUG:
            print(f"🚗 AI 진단 완료: Audit ID {event_data.audit_id}, Inspection ID {event_data.inspection_id}")
            print(f"📊 결과: {result['overall_grade'].value} (점수: {result['quality_score']:.3f})")
            print(f"⚠️ 결함 여부: {result['is_defect']}, 발견된 결함: {len(result['defects'])}개")
            for i, defect in enumerate(result["defects"], 1):
                print(f"  {i}. {defect.defect_type.value} - 신뢰도: {defect.confidence:.2f}, 심각도: {defect.severity:.2f}")
        
        return completed_event
        
    except Exception as e:
        print(f"❌ AI 진단 처리 실패: {e}")
        # 에러 발생 시에도 완료 이벤트 생성 (실패 상태로)
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
    """결과 파일 저장 경로 생성"""
    try:
        # 원본 파일 경로에서 디렉토리와 파일명 분리
        directory = os.path.dirname(collect_data_path)
        filename = os.path.basename(collect_data_path)
        name, ext = os.path.splitext(filename)
        
        # 결과 파일명 생성
        if error:
            result_filename = f"{name}_result_error_{inspection_id}.json"
        else:
            result_filename = f"{name}_result_{inspection_id}.json"
        
        # 결과 디렉토리 (results 하위 폴더)
        result_directory = os.path.join(directory, "results")
        os.makedirs(result_directory, exist_ok=True)
        
        return os.path.join(result_directory, result_filename)
        
    except Exception as e:
        # 기본 경로 사용
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"./results/diagnosis_result_{inspection_id}_{timestamp}.json"

def save_diagnosis_result(result_data: dict, file_path: str):
    """진단 결과를 파일로 저장"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 진단 결과 저장 완료: {file_path}")
        
    except Exception as e:
        print(f"❌ 결과 파일 저장 실패: {e}")
        raise

def get_model_status() -> dict:
    """모델 상태 조회"""
    return {
        "model_loaded": detector.model_loaded,
        "yolo_available": YOLO_AVAILABLE,
        "model_path": config.MODEL_PATH,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD
    }

# 레거시 호환을 위한 함수 (기존 API와의 호환성 유지)
def process_paint_inspection(request: PaintInspectionRequest) -> PaintInspectionResponse:
    """도장면 검사 처리 (기존 API 호환용)"""
    try:
        # YOLO 모델로 결함 검출
        result = detector.detect_defects(request.image_url)
        
        # 응답 객체 생성
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
        print(f"❌ 검사 처리 실패: {e}")
        raise Exception(f"검사 처리 실패: {e}")