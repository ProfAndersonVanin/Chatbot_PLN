# 🤖 Projeto: Chatbot de Atendimento com PLN

**Empresa fictícia:** TechStore Conecta — loja online de eletrônicos
**Ambiente:** Google Colab (gratuito)
**Duração:** 3 aulas (sugestão de ~100 min cada)
**Nível:** iniciante/intermediário em Python

---

## 🎯 O que vamos construir

Um assistente virtual que responde às dúvidas mais comuns dos clientes da TechStore Conecta: horário de atendimento, formas de pagamento, prazo de entrega, trocas, garantia e rastreamento de pedidos.

O chatbot funciona por **classificação de intenções**: o cliente escreve uma frase livre, o modelo descobre *o que ele quer* (a intenção) e o sistema devolve a resposta adequada.

```
Mensagem do cliente
        │
        ▼
┌──────────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐
│ Pré-processamento│ → │   TF-IDF     │ → │ Classificador     │ → │  Resposta    │
│ (limpeza, stems) │   │ (vetorização)│   │ (Reg. Logística)  │   │ + contexto   │
└──────────────────┘   └──────────────┘   └───────────────────┘   └──────────────┘
```

## 🗂️ Estrutura das aulas

| Aula | Arquivo | Tema | Entrega |
|------|---------|------|---------|
| 1 | [`aula01_base_e_preprocessamento.md`](aula01_base_e_preprocessamento.md) | Base de conhecimento e pré-processamento de texto | `base_conhecimento.json` + `preprocessamento.py` |
| 2 | [`aula02_classificacao_de_intencoes.md`](aula02_classificacao_de_intencoes.md) | Vetorização TF-IDF e classificação de intenções | `modelo_intencoes.joblib` |
| 3 | [`aula03_chatbot_e_interface.md`](aula03_chatbot_e_interface.md) | Motor de respostas, contexto, entidades e interface web | Chatbot funcionando no navegador |

## 🧰 Recursos utilizados (todos gratuitos)

| Recurso | Para quê |
|---------|----------|
| Google Colab | Executar os notebooks sem instalar nada |
| Google Drive | Guardar os arquivos entre uma aula e outra |
| NLTK | Stopwords e stemming em português (RSLP) |
| scikit-learn | TF-IDF, classificadores e métricas |
| pandas / matplotlib | Exploração e gráficos |
| joblib | Salvar e carregar o modelo treinado |
| Gradio | Interface de chat no navegador com link público temporário |

## 📁 Estrutura de pastas no Google Drive

Ao final das três aulas, a pasta do projeto ficará assim:

```
MyDrive/
└── chatbot_pln/
    ├── base_conhecimento.json     ← Aula 1
    ├── preprocessamento.py        ← Aula 1
    ├── modelo_intencoes.joblib    ← Aula 2
    └── log_conversas.csv          ← Aula 3
```

> 💡 **Por que usar o Drive?** O Colab apaga os arquivos quando a sessão termina. Salvando no Drive, o aluno retoma o projeto de onde parou na aula seguinte.

## ✅ Pré-requisitos dos alunos

- Conta Google
- Python básico: variáveis, listas, dicionários, funções e laços
- Noções de pré-processamento de texto (desejável, mas revisado na Aula 1)

## 🧭 Competências trabalhadas

- Modelagem de um problema real de atendimento como tarefa de PLN
- Construção e curadoria de uma base de dados de treino
- Pré-processamento de texto em português
- Representação vetorial de texto (TF-IDF)
- Treino e avaliação de classificadores (acurácia, precisão, revocação, F1, matriz de confusão)
- Extração simples de entidades com expressões regulares
- Gerenciamento de contexto de conversa
- Ciclo de melhoria contínua a partir de logs reais

## 📝 Sugestão de avaliação

| Critério | Peso |
|----------|------|
| Base de conhecimento ampliada (novas intenções e exemplos) | 25% |
| Modelo treinado e avaliado (relatório de métricas comentado) | 25% |
| Chatbot funcional na interface Gradio | 30% |
| Relatório curto: limitações encontradas e melhorias propostas | 20% |

**Projeto final sugerido:** cada grupo escolhe outra empresa (pizzaria, academia, clínica veterinária, secretaria da escola…) e adapta o chatbot, criando no mínimo 8 intenções com 10 exemplos cada.
