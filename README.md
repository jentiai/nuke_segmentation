# nuke_segmentation
합성데이터 전처리 + 세그멘테이션 모델 학습


### scripts/preprocess_synthetic_data.py

합성데이터를 키넥트v2 사용하던 당시 데이터 포맷으로 전처리하는 스크립트입니다.


### scripts/train.py

세그멘테이션 모델 학습 스크립트입니다.


### scripts/inference_2d.py

2D 추론 스크립트입니다. DataIO 의존 없이 자유롭게 경로를 입력해서 2D 단일 이미지에 대해 추론해보고 싶을 때 유용합니다.
스페이스바로 시맨틱 이미지 <===> RGB 이미지 간 전환할 수 있습니다.


### scripts/inference_3d.py

키넥트v2 사용하던 당시 데이터 포맷으로 이루어진 3D Scene을 추론하는 스크립트입니다.
&#91;와 	&#93;로 시맨틱 이미지 <===> RGB 이미지 간 전환할 수 있습니다.







젠티와 함께할 수 있어서 영광이었습니다.



감사합니다.
