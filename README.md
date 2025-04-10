# Classificador de Piscinas com Streamlit

Este projeto permite classificar imagens de satélite como contendo ou não piscinas. Inclui:

- Interface em Streamlit para upload e rotulagem
- Treinamento de modelo CNN com TensorFlow
- Classificação de novas imagens
- Visualização de desempenho

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estrutura de diretórios

- `data/positive/`: imagens com piscina
- `data/negative/`: imagens sem piscina
