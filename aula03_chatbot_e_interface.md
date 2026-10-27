# Aula 3 — Motor de Respostas, Contexto e Interface Web

**Projeto:** Chatbot de Atendimento TechStore Conecta
**Duração sugerida:** ~100 min
**Entrega da aula:** chatbot funcionando no navegador + `log_conversas.csv`

---

## 🎯 Objetivos

Ao final desta aula, o aluno será capaz de:

1. Integrar o modelo treinado a um **motor de respostas**.
2. Extrair uma **entidade** (número do pedido) com expressões regulares.
3. Manter um **contexto** simples de conversa (o bot "lembra" que pediu uma informação).
4. Publicar o chatbot em uma **interface web** gratuita com Gradio.
5. Usar **logs** para identificar falhas e melhorar o bot continuamente.

## ⏱️ Roteiro da aula

| Tempo | Atividade |
|-------|-----------|
| 0–10 min | Carregar base, módulo e modelo |
| 10–30 min | Motor de respostas e fallback |
| 30–50 min | Entidades e contexto: rastreamento de pedido |
| 50–60 min | Log de conversas |
| 60–80 min | Interface web com Gradio |
| 80–100 min | Ciclo de melhoria, apresentação e checklist |

---

## ⚙️ Parte 1 — Carregando tudo (10 min)

Crie um notebook `aula03_chatbot.ipynb`.

```python
# Célula 1 — Instalar o Gradio (interface web)
!pip install -q gradio
```

```python
# Célula 2 — Conectar ao Drive e carregar base, módulo e modelo
from google.colab import drive
drive.mount('/content/drive')

import sys, json, random, re
import joblib
import pandas as pd
from datetime import datetime

PASTA = '/content/drive/MyDrive/chatbot_pln'
sys.path.append(PASTA)
from preprocessamento import preprocessar

with open(f'{PASTA}/base_conhecimento.json', encoding='utf-8') as f:
    base = json.load(f)

modelo = joblib.load(f'{PASTA}/modelo_intencoes.joblib')

# Dicionário: intenção -> lista de respostas
respostas = {i['tag']: i['respostas'] for i in base['intencoes']}
print('Modelo carregado. Intenções conhecidas:', list(modelo.classes_))
```

---

## 💬 Parte 2 — Motor de respostas (20 min)

```python
# Célula 3 — Classificação com limiar (mesma lógica da Aula 2)
LIMIAR = 0.35

def classificar(mensagem):
    texto = preprocessar(mensagem)
    if not texto:
        return None, 0.0
    probs = modelo.predict_proba([texto])[0]
    indice = probs.argmax()
    confianca = float(probs[indice])
    if confianca < LIMIAR:
        return None, confianca
    return modelo.classes_[indice], confianca
```

```python
# Célula 4 — Primeira versão do bot
RESPOSTA_PADRAO = ("Desculpe, não entendi. 😅 Posso ajudar com: horário de atendimento, "
                   "formas de pagamento, entrega, rastreamento de pedido, trocas e garantia. "
                   "Se preferir, digite 'atendente'.")

def responder_v1(mensagem):
    intencao, confianca = classificar(mensagem)
    if intencao is None:
        return RESPOSTA_PADRAO
    return random.choice(respostas[intencao])

for frase in ["oi!", "aceitam pix?", "vocês vendem pizza?"]:
    print('Cliente:', frase)
    print('Bot:    ', responder_v1(frase), '\n')
```

> 💡 **Resposta de fallback:** em vez de um simples "não entendi", o bot **orienta** o cliente sobre o que sabe fazer. Isso melhora muito a experiência do usuário.

---

## 🔎 Parte 3 — Entidades e contexto (20 min)

### O problema

```
Cliente: quero rastrear meu pedido
Bot:     Qual é o número do pedido?
Cliente: 123456
Bot:     ??? (o modelo não sabe classificar "123456")
```

Precisamos de duas coisas:

1. **Extração de entidade:** encontrar o número do pedido no texto (regex).
2. **Contexto:** lembrar que o bot está **aguardando** um número de pedido.

```python
# Célula 5 — "Banco de dados" simulado de pedidos e extração de entidade
PEDIDOS = {
    '123456': '🚚 Em transporte — previsão de entrega em 2 dias úteis.',
    '654321': '✅ Entregue em 20/10.',
    '111222': '⏳ Pagamento em análise.',
}

def extrair_numero_pedido(mensagem):
    """Procura uma sequência de exatamente 6 dígitos."""
    encontrado = re.search(r'\b\d{6}\b', mensagem)
    return encontrado.group() if encontrado else None

def consultar_pedido(numero):
    status = PEDIDOS.get(numero)
    if status:
        return f'Pedido {numero}: {status}'
    return f'Não encontrei o pedido {numero}. Confira o número no e-mail de confirmação da compra.'

print(extrair_numero_pedido('status do pedido 123456, por favor'))
print(extrair_numero_pedido('meu pedido não chegou'))
```

```python
# Célula 6 — Log de conversas (para melhoria contínua)
log = []

def registrar(mensagem, intencao, confianca, resposta):
    log.append({
        'data_hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mensagem': mensagem,
        'intencao': intencao,
        'confianca': round(confianca, 3),
        'resposta': resposta,
    })
```

```python
# Célula 7 — Bot completo: respostas + entidade + contexto + log
contexto = {'aguardando_pedido': False}

def responder(mensagem):
    numero = extrair_numero_pedido(mensagem)

    # 1) O bot estava esperando o número do pedido?
    if contexto['aguardando_pedido']:
        contexto['aguardando_pedido'] = False
        if numero:
            resposta = consultar_pedido(numero)
            registrar(mensagem, 'rastrear_pedido', 1.0, resposta)
            return resposta
        # se não veio número, o cliente mudou de assunto: segue o fluxo normal

    # 2) Classificar a intenção
    intencao, confianca = classificar(mensagem)

    if intencao is None:
        resposta = RESPOSTA_PADRAO
    elif intencao == 'rastrear_pedido':
        if numero:
            resposta = consultar_pedido(numero)
        else:
            contexto['aguardando_pedido'] = True
            resposta = random.choice(respostas[intencao])
    else:
        resposta = random.choice(respostas[intencao])

    registrar(mensagem, intencao, confianca, resposta)
    return resposta
```

```python
# Célula 8 — Testar uma conversa completa
conversa = [
    "boa noite",
    "quero rastrear meu pedido",
    "123456",
    "e o pedido 999999?",
    "aceita pix?",
    "vocês consertam geladeira?",
    "valeu, tchau",
]
for msg in conversa:
    print(f'🧑 {msg}')
    print(f'🤖 {responder(msg)}\n')
```

> 🗣️ **Discussão:** o `contexto` é uma variável global. O que aconteceria se **dois clientes** usassem o bot ao mesmo tempo? (Em sistemas reais, o contexto é guardado **por sessão/usuário**.)

### Testando no terminal do Colab (opcional)

```python
# Célula 9 — Conversa interativa (digite 'sair' para encerrar)
while True:
    msg = input('Você: ')
    if msg.lower().strip() == 'sair':
        break
    print('Bot:', responder(msg))
```

---

## 🌐 Parte 4 — Interface web com Gradio (20 min)

O Gradio cria uma janela de chat com poucas linhas de código e gera um **link público temporário** que pode ser aberto até no celular.

```python
# Célula 10 — Interface de chat
import gradio as gr

def chat_gradio(mensagem, historico):
    return responder(mensagem)

demo = gr.ChatInterface(
    fn=chat_gradio,
    title='🛒 TechStore Conecta — Assistente Virtual',
    description='Tire suas dúvidas sobre pagamento, entrega, trocas, garantia e pedidos.',
    examples=['Quais as formas de pagamento?',
              'Quero rastrear meu pedido',
              'Qual a garantia dos produtos?',
              'Quero falar com um atendente'],
)

demo.launch(share=True)
```

> 📱 **Atividade:** cada aluno abre o link `...gradio.live` no celular e conversa com o bot do colega, tentando "confundi-lo". Todas as conversas ficam registradas no log!

> ⚠️ O link público é **temporário** e funciona apenas enquanto o notebook estiver em execução. Para interromper, clique no botão ⏹ da célula ou rode `demo.close()`.

---

## 🔁 Parte 5 — Ciclo de melhoria contínua (15 min)

```python
# Célula 11 — Salvar e analisar o log
df_log = pd.DataFrame(log)
df_log.to_csv(f'{PASTA}/log_conversas.csv', index=False, encoding='utf-8')

print('Mensagens registradas:', len(df_log))
print('Mensagens NÃO entendidas:', df_log['intencao'].isna().sum())
```

```python
# Célula 12 — O que o bot não entendeu ou entendeu com pouca confiança?
df_log[(df_log['intencao'].isna()) | (df_log['confianca'] < 0.5)][['mensagem', 'intencao', 'confianca']]
```

O ciclo que empresas reais utilizam:

```
  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
  │ Clientes usam│ ──► │ Log registra │ ──► │ Equipe analisa  │ ──► │ Base ampliada│
  │   o chatbot  │     │  as falhas   │     │ e rotula frases │     │ e retreino   │
  └──────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
          ▲                                                                  │
          └──────────────────────────────────────────────────────────────────┘
```

> 🧪 **Atividade:** escolha 5 mensagens do log que falharam, adicione-as à `base_conhecimento.json` na intenção correta (ou crie uma nova intenção), rode novamente a Aula 2 e verifique se o bot passou a acertar.

---

## 🚀 Desafios extras (para quem terminar antes)

### Desafio A — Busca semântica com embeddings (gratuito)

Modelos de *sentence embeddings* entendem que *"o aparelho pifou"* e *"notebook parou de funcionar"* têm sentido parecido, mesmo sem palavras em comum.

```python
!pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer, util

emb_modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

exemplos, tags = [], []
for i in base['intencoes']:
    for e in i['exemplos']:
        exemplos.append(e)
        tags.append(i['tag'])

emb_base = emb_modelo.encode(exemplos, convert_to_tensor=True)

def classificar_semantico(mensagem, limiar=0.5):
    emb = emb_modelo.encode(mensagem, convert_to_tensor=True)
    sims = util.cos_sim(emb, emb_base)[0]
    idx = int(sims.argmax())
    score = float(sims[idx])
    return (tags[idx] if score >= limiar else None), score

for f in ["o aparelho pifou", "dá pra pagar com pix?", "vocês vendem pizza?"]:
    print(f, '->', classificar_semantico(f))
```

> 🗣️ Compare com o modelo TF-IDF: em que frases cada abordagem se sai melhor? Qual é mais leve e rápida?

### Desafio B — Contexto por usuário

Use o parâmetro `historico` da função `chat_gradio` para descobrir se a **última resposta do bot** pediu o número do pedido, eliminando a variável global `contexto`.

### Desafio C — Publicação permanente e gratuita

Publique o bot no **Hugging Face Spaces** (plano gratuito): crie um Space do tipo Gradio e envie os arquivos `app.py` (com o código das Células 2 a 10, sem a parte do Drive), `preprocessamento.py`, `base_conhecimento.json`, `modelo_intencoes.joblib` e um `requirements.txt` contendo:

```
gradio
scikit-learn
nltk
joblib
```

> ⚠️ Use no `requirements.txt` a mesma versão do scikit-learn do Colab (`import sklearn; print(sklearn.__version__)`) para evitar incompatibilidade ao carregar o modelo.

---

## 🧭 Fechamento e reflexão

Discuta com a turma:

1. Quais são as **limitações** desse chatbot? (Só responde ao que está na base, não entende frases longas com várias perguntas, contexto simples…)
2. Em que situações uma empresa **prefere** um bot de intenções a um bot generativo? (Controle total das respostas, custo zero de API, sem risco de inventar políticas da empresa.)
3. Que cuidados de **privacidade (LGPD)** são necessários ao guardar logs de conversas de clientes reais?

---

## ✅ Checklist final do projeto

- [ ] Bot responde corretamente às 10 intenções da base
- [ ] Bot mostra uma resposta de fallback útil quando não entende
- [ ] Fluxo de rastreamento funciona com o número na mesma frase **e** em mensagem separada
- [ ] Interface Gradio funcionando com link público
- [ ] `log_conversas.csv` salvo e analisado
- [ ] Pelo menos uma rodada do ciclo de melhoria realizada (base ampliada + retreino)

🎉 **Parabéns!** Você construiu um chatbot de atendimento completo usando apenas ferramentas gratuitas e técnicas clássicas de PLN.
