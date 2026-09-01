"""나무위키 분산 수집·RAG 검색 파이프라인의 공용 라이브러리.

각 서비스(crawler·parser·embedder·api)가 공유하는 조각들이 여기 모여 있다.
서비스끼리는 직접 import 하지 않고 Kafka 토픽으로만 통신한다.
"""

__version__ = "0.1.0"
