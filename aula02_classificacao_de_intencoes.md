# Aula 2 — Vetorização TF-IDF e Classificação de Intenções

**Projeto:** Chatbot de Atendimento TechStore Conecta
**Duração sugerida:** ~100 min
**Entrega da aula:** `modelo_intencoes.joblib` salvo no Google Drive

---

## 🎯 Objetivos

Ao final desta aula, o aluno será capaz de:

1. Explicar por que modelos de Machine Learning precisam de texto convertido em números.
2. Aplicar a vetorização **TF-IDF** e interpretar a matriz gerada.
3. Treinar e comparar classificadores de intenção com scikit-learn.
4. Avaliar o modelo com acurácia, precisão, revocação, F1 e matriz de confusão.
5. Definir um **limiar de confiança** para que o bot saiba dizer "não entendi".

## ⏱️ Roteiro da aula

| Tempo | Atividade |
|-------|-----------|
| 0–10 min | Revisão da Aula 1 e carregamento dos arquivos |
| 10–30 min | Conceito e prática de TF-IDF |
| 30–55 min | Treino e comparação de classificadores |
| 55–70 min | Avaliação: métricas e matriz de confusão |
| 70–85 min | Limiar de confiança e testes com frases novas |
| 85–100 min | Salvar o modelo, exercícios e checklist |

---

## ⚙️ Parte 1 — Carregando o trabalho da Aula 1 (10 min)

Crie um notebook `aula02_chatbot.ipynb`.

```python
# Célula 1 — Conectar ao Drive e importar o módulo da Aula 1
from google.colab import drive
drive.mount('/content/drive')

import sys, json
PASTA = '/content/drive/MyDrive/chatbot_pln'
sys.path.append(PASTA)

from preprocessamento import preprocessar

with open(f'{PASTA}/base_conhecimento.json', encoding='utf-8') as f:
    base = json.load(f)

print('Empresa:', base['empresa'])
print('Intenções:', [i['tag'] for i in base['intencoes']])
```

```python
# Célula 2 — Montar os dados de treino (X = textos, y = intenções)
import pandas as pd

textos, rotulos = [], []
for intencao in base['intencoes']:
    for exemplo in intencao['exemplos']:
        textos.append(exemplo)
        rotulos.append(intencao['tag'])

df = pd.DataFrame({'texto': textos, 'intencao': rotulos})
df['texto_limpo'] = df['texto'].apply(preprocessar)
print('Total de exemplos:', len(df))
df.head()
```

---

## 🔢 Parte 2 — TF-IDF: transformando texto em números (20 min)

### Conceito

Um modelo de ML só trabalha com números. O **TF-IDF** atribui um peso para cada palavra em cada frase:

- **TF (Term Frequency):** quantas vezes a palavra aparece **na frase**.
- **IDF (Inverse Document Frequency):** quão **rara** a palavra é no conjunto inteiro.

> 💡 **Intuição:** a palavra `pix` aparece em poucas frases e é muito informativa → peso alto. Uma palavra que aparece em quase todas as frases diz pouco sobre a intenção → peso baixo.

Fórmula simplificada:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$$

onde *N* é o número de frases e *DF(t)* é o número de frases que contêm o termo *t*.

### Prática

```python
# Célula 3 — Vetorizar um pequeno exemplo para visualizar
from sklearn.feature_extraction.text import TfidfVectorizer

mini = ["aceita pix", "desconto no pix", "prazo de entrega", "prazo para troca"]
mini_limpo = [preprocessar(t) for t in mini]

vet_demo = TfidfVectorizer()
matriz = vet_demo.fit_transform(mini_limpo)

pd.DataFrame(matriz.toarray().round(2),
             columns=vet_demo.get_feature_names_out(),
             index=mini)
```

> 🗣️ **Discussão:** na frase "prazo de entrega", por que `entreg` recebe peso maior do que `praz`? (Dica: em quantas frases cada termo aparece?) E por que a palavra "prazo" sozinha não basta para decidir entre `prazo_entrega` e `troca_devolucao`?

### N-gramas

Com `ngram_range=(1, 2)`, o TF-IDF considera também **pares de palavras** (bigramas), como `nao cheg`. Isso ajuda o modelo a capturar expressões.

```python
# Célula 4 — Vetorizar a base completa com unigramas e bigramas
vetorizador = TfidfVectorizer(ngram_range=(1, 2))
X_todo = vetorizador.fit_transform(df['texto_limpo'])
print('Formato da matriz:', X_todo.shape, '(frases x termos)')
print('Alguns termos:', vetorizador.get_feature_names_out()[:20])
```

---

## 🧠 Parte 3 — Treinando classificadores (25 min)

### Separando treino e teste

Nunca avaliamos o modelo com as mesmas frases que ele usou para aprender. É como aplicar uma prova com as mesmas questões da lista de exercícios.

```python
# Célula 5 — Separar treino (75%) e teste (25%)
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(
    df['texto_limpo'], df['intencao'],
    test_size=0.25, random_state=42, stratify=df['intencao']
)
print('Treino:', len(X_treino), '| Teste:', len(X_teste))
```

> 📌 `stratify` garante que todas as intenções apareçam no treino e no teste na mesma proporção.

### Pipeline

Um `Pipeline` encadeia vetorização + classificador em um único objeto. Assim, o mesmo vetorizador treinado é usado sempre, evitando erros.

```python
# Célula 6 — Comparar três classificadores
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelos = {
    'Regressão Logística': LogisticRegression(max_iter=1000, C=10),
    'Naive Bayes': MultinomialNB(),
    'SVM Linear': LinearSVC(),
}

resultados = []
for nome, clf in modelos.items():
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('clf', clf)])
    pipe.fit(X_treino, y_treino)
    acc = accuracy_score(y_teste, pipe.predict(X_teste))
    resultados.append({'modelo': nome, 'acuracia': round(acc, 3)})

pd.DataFrame(resultados).sort_values('acuracia', ascending=False)
```

> 💡 **Por que vamos seguir com a Regressão Logística?** Além de ter bom desempenho com textos curtos, ela fornece **probabilidades** (`predict_proba`), que usaremos para medir a confiança do bot. O `LinearSVC` não oferece isso diretamente.

> ⚠️ Com uma base pequena (~80 frases), os resultados variam bastante a cada divisão. Em projetos reais, usamos **validação cruzada** (ver Exercício 3).

---

## 📊 Parte 4 — Avaliando o modelo (15 min)

```python
# Célula 7 — Treinar o modelo escolhido e gerar o relatório
from sklearn.metrics import classification_report

modelo = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, C=10))
])
modelo.fit(X_treino, y_treino)
y_pred = modelo.predict(X_teste)

print(classification_report(y_teste, y_pred, zero_division=0))
```

| Métrica | Pergunta que responde |
|---------|-----------------------|
| **Precisão** | Das vezes que o bot disse "é `garantia`", quantas estavam certas? |
| **Revocação (recall)** | Das frases que eram realmente `garantia`, quantas o bot reconheceu? |
| **F1-score** | Equilíbrio entre precisão e revocação |
| **Acurácia** | Percentual total de acertos |

```python
# Célula 8 — Matriz de confusão
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay.from_predictions(y_teste, y_pred, ax=ax,
                                        xticks_rotation=45, cmap='Blues', colorbar=False)
plt.title('Matriz de confusão — intenções')
plt.tight_layout()
plt.show()
```

```python
# Célula 9 — Ver exatamente onde o modelo errou
erros = pd.DataFrame({'texto': X_teste, 'real': y_teste, 'previsto': y_pred})
erros[erros['real'] != erros['previsto']]
```

> 🗣️ **Discussão:** as confusões fazem sentido? Ex.: "quero falar com o suporte" (`falar_atendente`) pode ser confundida com "até que horas funciona o suporte?" (`horario_atendimento`). Como a base poderia ser melhorada?

---

## 🎚️ Parte 5 — Limiar de confiança (15 min)

Um classificador **sempre** escolhe uma das intenções, mesmo para frases sem sentido como *"qual a capital da França?"*. Precisamos que o bot saiba admitir quando não entendeu.

```python
# Célula 10 — Treinar o modelo final com TODOS os dados
modelo_final = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, C=10))
])
modelo_final.fit(df['texto_limpo'], df['intencao'])
```

> 📌 Depois de avaliado, o modelo é retreinado com **100% dos dados**, pois em produção queremos que ele aprenda com tudo o que temos.

```python
# Célula 11 — Função de classificação com limiar de confiança
LIMIAR = 0.35

def classificar(mensagem, modelo=modelo_final, limiar=LIMIAR):
    texto = preprocessar(mensagem)
    if not texto:                         # frase vazia após a limpeza
        return None, 0.0
    probs = modelo.predict_proba([texto])[0]
    indice = probs.argmax()
    intencao = modelo.classes_[indice]
    confianca = probs[indice]
    if confianca < limiar:
        return None, confianca
    return intencao, confianca

frases_novas = [
    "boa tarde, tudo certo?",
    "dá pra pagar com pix?",
    "minha compra ainda não chegou",
    "o fone veio quebrado, quero trocar",
    "quanto tempo dura a garantia do notebook?",
    "qual a capital da França?",
    "vocês vendem pizza?",
]

for frase in frases_novas:
    intencao, conf = classificar(frase)
    print(f'{frase:45} -> {str(intencao):20} ({conf:.0%})')
```

```python
# Célula 12 — Ver as 3 intenções mais prováveis de uma frase
def top3(mensagem):
    probs = modelo_final.predict_proba([preprocessar(mensagem)])[0]
    ranking = sorted(zip(modelo_final.classes_, probs), key=lambda x: -x[1])[:3]
    for tag, p in ranking:
        print(f'  {tag:20} {p:.0%}')

top3("vocês vendem pizza?")
```

> 🧪 **Atividade:** testem limiares entre 0.20 e 0.60. Limiar **alto** → o bot diz "não entendi" com frequência. Limiar **baixo** → o bot responde coisas erradas com confiança. Qual equilíbrio a turma escolhe?

---

## 💾 Parte 6 — Salvando o modelo (5 min)

```python
# Célula 13 — Salvar o modelo treinado no Drive
import joblib

joblib.dump(modelo_final, f'{PASTA}/modelo_intencoes.joblib')
print('Modelo salvo em', f'{PASTA}/modelo_intencoes.joblib')
```

---

## 🏋️ Exercícios

1. **Robustez:** escreva 10 frases novas (que não estão na base) e registre quantas o modelo acertou. Anote as que errou.
2. **Melhoria guiada por erro:** adicione as frases que o modelo errou à `base_conhecimento.json` (na intenção correta), retreine e compare a acurácia.
3. **Validação cruzada:** use `cross_val_score(modelo, df['texto_limpo'], df['intencao'], cv=4)` e compare a média com a acurácia da Célula 7. Por que esse resultado é mais confiável?
4. **Experimento:** retreine usando `preprocessar(t, usar_stemming=False)`. O desempenho melhora ou piora?
5. **Desafio:** troque `TfidfVectorizer(ngram_range=(1, 2))` por `TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))` (n-gramas de **caracteres**) e teste com erros de digitação, como `"garntia"` e `"pagamneto"`.

---

## ✅ Checklist de saída

- [ ] Sei explicar o que o TF-IDF faz e por que palavras raras pesam mais
- [ ] Comparei pelo menos três classificadores
- [ ] Interpretei o relatório de classificação e a matriz de confusão
- [ ] Defini um limiar de confiança e justifiquei a escolha
- [ ] Arquivo `modelo_intencoes.joblib` salvo no Drive

**Próxima aula:** vamos dar voz ao modelo — respostas, contexto de conversa, extração do número do pedido e uma interface web de chat.
