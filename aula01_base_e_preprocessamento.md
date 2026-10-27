# Aula 1 — Base de Conhecimento e Pré-processamento

**Projeto:** Chatbot de Atendimento TechStore Conecta
**Duração sugerida:** ~100 min
**Entregas da aula:** `base_conhecimento.json` e `preprocessamento.py` salvos no Google Drive

---

## 🎯 Objetivos

Ao final desta aula, o aluno será capaz de:

1. Explicar o que é um chatbot baseado em **intenções** e em que ele difere de um chatbot generativo (como o ChatGPT ou o Claude).
2. Montar uma **base de conhecimento** com intenções, exemplos de frases e respostas.
3. Escrever uma função de **pré-processamento** de texto em português.
4. Organizar o código em um módulo `.py` reutilizável nas próximas aulas.

## ⏱️ Roteiro da aula

| Tempo | Atividade |
|-------|-----------|
| 0–15 min | Conceitos: tipos de chatbot e o problema da TechStore |
| 15–25 min | Configuração do Colab e do Google Drive |
| 25–50 min | Construção da base de conhecimento |
| 50–60 min | Exploração da base com pandas |
| 60–90 min | Pré-processamento e criação do módulo |
| 90–100 min | Exercícios e checklist |

---

## 📚 Parte 1 — Conceitos (15 min)

### Tipos de chatbot

| Tipo | Como funciona | Vantagens | Limitações |
|------|---------------|-----------|------------|
| **Regras** (if/else, palavras-chave) | Procura palavras exatas | Simples, previsível | Quebra com sinônimos e erros de digitação |
| **Classificação de intenções** ← *nosso projeto* | Um modelo de ML descobre a intenção da frase | Leve, barato, controlável, roda em qualquer máquina | Só responde ao que foi previsto na base |
| **Generativo** (LLMs) | Gera o texto da resposta | Muito flexível | Custo, pode "inventar" informações, exige mais cuidado |

### Vocabulário do projeto

- **Intenção (intent):** o objetivo do cliente. Ex.: `formas_pagamento`.
- **Exemplos (patterns):** frases diferentes que expressam a mesma intenção. Ex.: *"aceita pix?"*, *"posso pagar no boleto?"*.
- **Respostas:** o que o bot diz quando reconhece aquela intenção.
- **Entidade:** uma informação específica dentro da frase. Ex.: o número do pedido em *"status do pedido 123456"* (veremos na Aula 3).

### O problema da TechStore Conecta

A loja recebe centenas de mensagens por dia e a maioria se repete. A equipe levantou as 10 dúvidas mais frequentes — elas serão as nossas intenções.

> 🗣️ **Pergunta para a turma:** que outras dúvidas um cliente de loja online costuma ter?

---

## ⚙️ Parte 2 — Configurando o ambiente (10 min)

1. Acesse [colab.research.google.com](https://colab.research.google.com) e crie um notebook chamado `aula01_chatbot.ipynb`.
2. Execute a célula abaixo e autorize o acesso ao Drive.

```python
# Célula 1 — Conectar ao Google Drive e criar a pasta do projeto
from google.colab import drive
drive.mount('/content/drive')

import os
PASTA = '/content/drive/MyDrive/chatbot_pln'
os.makedirs(PASTA, exist_ok=True)
print('Pasta do projeto:', PASTA)
```

---

## 🧱 Parte 3 — Construindo a base de conhecimento (25 min)

Cada intenção tem uma `tag`, uma lista de `exemplos` e uma lista de `respostas`.

> 💡 **Dica didática:** peça para os alunos lerem os exemplos em voz alta e perceberem a variedade: formal, informal, com e sem acento, com gírias. Quanto mais variada a base, melhor o modelo generaliza.

```python
# Célula 2 — Base de conhecimento da TechStore Conecta
base = {
    "empresa": "TechStore Conecta",
    "intencoes": [
        {
            "tag": "saudacao",
            "exemplos": ["oi", "olá", "bom dia", "boa tarde", "boa noite",
                         "e aí, tudo bem?", "olá, preciso de ajuda", "oi, alguém aí?"],
            "respostas": ["Olá! Sou o assistente virtual da TechStore Conecta. Como posso ajudar?",
                          "Oi! Tudo bem? Em que posso te ajudar hoje?"]
        },
        {
            "tag": "despedida",
            "exemplos": ["tchau", "até logo", "até mais", "falou, tchau",
                         "encerrar atendimento", "pode encerrar", "flw", "até a próxima"],
            "respostas": ["Até logo! A TechStore Conecta agradece o contato. 👋",
                          "Tchau! Sempre que precisar, é só chamar."]
        },
        {
            "tag": "agradecimento",
            "exemplos": ["obrigado", "obrigada", "valeu", "muito obrigado pela ajuda",
                         "agradeço", "valeu mesmo", "show, obrigado", "ajudou muito"],
            "respostas": ["Por nada! Posso ajudar em mais alguma coisa?",
                          "Fico feliz em ajudar! 😊"]
        },
        {
            "tag": "horario_atendimento",
            "exemplos": ["qual o horário de atendimento?", "vocês abrem que horas?",
                         "até que horas funciona o suporte?", "atendem no sábado?",
                         "funcionam domingo?", "que horas fecha?",
                         "horário de funcionamento", "o atendimento é 24 horas?"],
            "respostas": ["Nosso atendimento humano funciona de segunda a sexta, das 8h às 20h, "
                          "e aos sábados, das 9h às 14h. Já este assistente virtual atende 24h!"]
        },
        {
            "tag": "formas_pagamento",
            "exemplos": ["quais as formas de pagamento?", "aceita pix?", "posso pagar no boleto?",
                         "parcela no cartão?", "em quantas vezes posso parcelar?",
                         "aceitam cartão de débito?", "tem desconto no pix?", "como posso pagar?"],
            "respostas": ["Aceitamos Pix (com 5% de desconto), boleto bancário e cartão de crédito "
                          "em até 10x sem juros."]
        },
        {
            "tag": "prazo_entrega",
            "exemplos": ["qual o prazo de entrega?", "quanto tempo demora para chegar?",
                         "vocês entregam em todo o brasil?", "o frete é grátis?",
                         "quanto custa o frete?", "em quantos dias chega?",
                         "fazem entrega expressa?", "entregam no mesmo dia?"],
            "respostas": ["Entregamos em todo o Brasil! O prazo varia de 2 a 10 dias úteis conforme a região. "
                          "O frete é grátis para compras acima de R$ 299."]
        },
        {
            "tag": "rastrear_pedido",
            "exemplos": ["onde está meu pedido?", "quero rastrear meu pedido", "meu pedido não chegou",
                         "status do pedido 123456", "como acompanho minha compra?",
                         "código de rastreio", "já enviaram meu pedido?", "minha encomenda está atrasada"],
            "respostas": ["Posso verificar para você! Qual é o número do pedido (6 dígitos)?"]
        },
        {
            "tag": "troca_devolucao",
            "exemplos": ["quero trocar um produto", "como faço devolução?", "produto veio com defeito",
                         "posso devolver?", "qual o prazo para troca?", "me arrependi da compra",
                         "veio o produto errado", "quero meu dinheiro de volta"],
            "respostas": ["Você pode solicitar troca ou devolução em até 7 dias após o recebimento, "
                          "pelo menu 'Meus Pedidos' no site. Produtos com defeito têm até 30 dias."]
        },
        {
            "tag": "garantia",
            "exemplos": ["qual a garantia dos produtos?", "o produto tem garantia?", "garantia estendida",
                         "meu celular estragou na garantia", "como acionar a garantia?",
                         "quanto tempo de garantia?", "a garantia é do fabricante?",
                         "notebook parou de funcionar"],
            "respostas": ["Todos os produtos têm garantia de 12 meses do fabricante. "
                          "Para acioná-la, envie a nota fiscal pelo e-mail garantia@techstoreconecta.com.br."]
        },
        {
            "tag": "falar_atendente",
            "exemplos": ["quero falar com um atendente", "falar com humano", "me transfere para uma pessoa",
                         "atendimento humano", "não quero falar com robô", "tem alguém real aí?",
                         "chama um atendente", "quero falar com o suporte"],
            "respostas": ["Certo! Vou te transferir para um atendente. Tempo médio de espera: 5 minutos."]
        }
    ]
}
print(f"Intenções cadastradas: {len(base['intencoes'])}")
```

```python
# Célula 3 — Salvar a base no Drive em formato JSON
import json

caminho_base = f'{PASTA}/base_conhecimento.json'
with open(caminho_base, 'w', encoding='utf-8') as f:
    json.dump(base, f, ensure_ascii=False, indent=2)

print('Base salva em:', caminho_base)
```

> 📌 **Por que JSON?** É legível por humanos, fácil de editar e é o formato usado por muitas plataformas de chatbot do mercado.

---

## 🔍 Parte 4 — Explorando a base (10 min)

```python
# Célula 4 — Transformar em tabela para analisar
import pandas as pd

linhas = []
for intencao in base['intencoes']:
    for exemplo in intencao['exemplos']:
        linhas.append({'texto': exemplo, 'intencao': intencao['tag']})

df = pd.DataFrame(linhas)
print('Total de exemplos:', len(df))
df.sample(10, random_state=42)
```

```python
# Célula 5 — Quantos exemplos por intenção?
import matplotlib.pyplot as plt

df['intencao'].value_counts().plot(kind='barh', figsize=(8, 4), color='steelblue')
plt.title('Exemplos por intenção')
plt.xlabel('Quantidade')
plt.tight_layout()
plt.show()
```

> 🗣️ **Discussão:** o que aconteceria se uma intenção tivesse 50 exemplos e outra apenas 2? (Conceito de **desbalanceamento de classes**.)

---

## 🧹 Parte 5 — Pré-processamento (30 min)

O computador não entende que *"Olá!"*, *"ola"* e *"OLÁ"* são a mesma coisa. O pré-processamento padroniza o texto antes de entregá-lo ao modelo.

| Etapa | Antes | Depois |
|-------|-------|--------|
| Minúsculas | `Quero TROCAR` | `quero trocar` |
| Remover acentos | `não`, `devolução` | `nao`, `devolucao` |
| Substituir números | `pedido 123456` | `pedido numero` |
| Tokenização | `quero trocar o produto` | `['quero', 'trocar', 'o', 'produto']` |
| Remover stopwords | `['quero', 'trocar', 'o', 'produto']` | `['quero', 'trocar', 'produto']` |
| Stemming (RSLP) | `['trocar', 'trocas', 'troca']` | `['troc', 'troc', 'troc']` |

### ⚠️ Cuidado com as stopwords!

A lista padrão de stopwords do português inclui palavras como **"não"**, **"quando"**, **"como"** e **"qual"**. Em um chatbot, elas carregam significado: *"meu pedido chegou"* ≠ *"meu pedido **não** chegou"*. Por isso vamos **mantê-las**.

### Criando o módulo reutilizável

Vamos salvar a função em um arquivo `.py` no Drive para importá-la nas Aulas 2 e 3, garantindo que o texto seja tratado **sempre da mesma forma** no treino e no uso real.

```python
%%writefile /content/drive/MyDrive/chatbot_pln/preprocessamento.py
# Célula 6 — Módulo de pré-processamento (salvo no Drive)
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('stopwords', quiet=True)
nltk.download('rslp', quiet=True)


def remover_acentos(texto):
    """Remove acentos: 'devolução' -> 'devolucao'."""
    nfkd = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# Palavras que NÃO queremos remover, pois mudam o sentido da frase
PALAVRAS_IMPORTANTES = {'nao', 'como', 'quando', 'onde', 'qual', 'quais',
                        'quanto', 'quantos', 'quantas', 'que'}

STOPWORDS = {remover_acentos(p) for p in stopwords.words('portuguese')} - PALAVRAS_IMPORTANTES
STEMMER = RSLPStemmer()


def preprocessar(texto, usar_stemming=True):
    """Padroniza um texto para ser usado pelo modelo."""
    texto = texto.lower()
    texto = remover_acentos(texto)
    texto = re.sub(r'\d+', ' numero ', texto)       # números viram o token 'numero'
    tokens = re.findall(r'[a-z]+', texto)            # tokenização simples
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if usar_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]
    return ' '.join(tokens)
```

> ⚠️ A linha `%%writefile` precisa ser obrigatoriamente a **primeira linha** da célula; caso contrário, o Colab tenta executar o código em vez de salvá-lo.

```python
# Célula 7 — Importar o módulo e testar
import sys
sys.path.append(PASTA)

from preprocessamento import preprocessar

frases_teste = [
    "Olá, bom dia!",
    "Meu pedido 123456 NÃO chegou ainda...",
    "Quais são as formas de pagamento?",
    "Quero trocar o produto, veio com defeito",
    "Vocês entregam em Ribeirão Pires?"
]

for frase in frases_teste:
    print(f'{frase:45} -> {preprocessar(frase)}')
```

```python
# Célula 8 — Comparar com e sem stemming na base inteira
df['sem_stemming'] = df['texto'].apply(lambda t: preprocessar(t, usar_stemming=False))
df['com_stemming'] = df['texto'].apply(preprocessar)
df.head(15)
```

```python
# Célula 9 — Tamanho do vocabulário
vocab_sem = set(' '.join(df['sem_stemming']).split())
vocab_com = set(' '.join(df['com_stemming']).split())
print('Vocabulário sem stemming:', len(vocab_sem))
print('Vocabulário com stemming:', len(vocab_com))
```

> 🗣️ **Discussão:** por que um vocabulário menor pode ajudar quando temos **poucos dados**? E qual é o risco de reduzir demais? (Ex.: palavras diferentes que viram o mesmo radical.)

---

## 🏋️ Exercícios

1. **Ampliar a base:** adicione pelo menos 4 novos exemplos em 3 intenções diferentes, usando gírias, abreviações (`vcs`, `pq`, `qnd`) ou erros de digitação comuns.
2. **Nova intenção:** crie a intenção `cupom_desconto` com 8 exemplos e 2 respostas. Salve a base novamente (Célula 3).
3. **Investigação:** teste `preprocessar("não quero falar com robô")`. O que acontece com a palavra "não"? E se você remover `'nao'` de `PALAVRAS_IMPORTANTES`?
4. **Caso curioso:** rode `preprocessar("até mais")`. Por que o resultado fica vazio? Que problema isso causa para o modelo e como você resolveria? (Dica: `PALAVRAS_IMPORTANTES`.)
5. **Desafio:** e se o cliente digitar `"vcs aceitam pix?"`? Crie um dicionário de abreviações (`{'vcs': 'voces', 'pq': 'porque', ...}`) e aplique-o dentro de `preprocessar`, antes da remoção de stopwords.

---

## ✅ Checklist de saída

- [ ] Pasta `chatbot_pln` criada no Google Drive
- [ ] Arquivo `base_conhecimento.json` salvo com, no mínimo, 10 intenções
- [ ] Arquivo `preprocessamento.py` salvo e importando sem erros
- [ ] Sei explicar por que mantivemos palavras como "não" e "quando"

**Próxima aula:** vamos transformar texto em números (TF-IDF) e treinar o modelo que reconhece as intenções.
