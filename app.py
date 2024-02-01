Input (224x224x3)
|
|-- Conv (3x3, 64) + ReLU
|-- Conv (3x3, 64) + ReLU
|-- Max Pooling (2x2)
|
|-- Conv (3x3, 128) + ReLU
|-- Conv (3x3, 128) + ReLU
|-- Max Pooling (2x2)
|
|-- Conv (3x3, 256) + ReLU
|-- Conv (3x3, 256) + ReLU
|-- Conv (3x3, 256) + ReLU
|-- Max Pooling (2x2)
|
|-- Conv (3x3, 512) + ReLU
|-- Conv (3x3, 512) + ReLU
|-- Conv (3x3, 512) + ReLU
|-- Max Pooling (2x2)
|
|-- Conv (3x3, 512) + ReLU
|-- Conv (3x3, 512) + ReLU
|-- Conv (3x3, 512) + ReLU
|-- Max Pooling (2x2)
|
|-- Fully Connected (4096) + ReLU
|-- Fully Connected (4096) + ReLU
|-- Fully Connected (1000) + Softmax (Output)
