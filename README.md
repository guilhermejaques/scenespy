# Scenespy

Scenespy é um app desktop para separar vídeos em partes automaticamente. Ele pode detectar mudanças de cena, cortar por intervalo fixo de tempo ou extrair rostos encontrados no vídeo.

O objetivo é simples: escolher um vídeo, escolher uma pasta de saída, selecionar o modo de corte e deixar o app processar.

## Imagens do app

Substitua estes espaços pelas suas capturas de tela:

![Tela principal](docs/images/main-window.png)

![Processamento em andamento](docs/images/processing.png)

![Resultado gerado](docs/images/output-example.png)

## O que ele faz

- Detecta cortes de cena em vídeos.
- Divide vídeos em segmentos por intervalo de segundos.
- Detecta rostos e salva imagens dos melhores recortes encontrados.
- Processa um ou vários vídeos em fila.
- Mostra progresso, tempo estimado e prévia visual.
- Cria uma pasta de saída organizada para cada processamento.
- Gera arquivos de metadata como `scenes.json` e, quando necessário, `cut_errors.json`.

## Modos disponíveis

### Scene detection

Analisa o vídeo e tenta encontrar mudanças naturais de cena. É útil para filmes, séries, trailers, vídeos de gameplay, aulas editadas e conteúdos com cortes visuais.

### Every seconds

Corta o vídeo em partes com duração fixa. É o modo mais previsível: você escolhe o intervalo em segundos e o app divide o vídeo.

### Detect faces

Procura rostos no vídeo e salva imagens dos rostos detectados. Esse modo usa dependências de IA e pode ser mais pesado.

## Como usar

1. Abra o Scenespy.
2. Em **Source video(s)**, selecione um ou mais vídeos.
3. Em **Output folder**, escolha onde os arquivos serão salvos.
4. Escolha o modo:
   - **Scene detection**
   - **Every seconds**
   - **Detect faces**
5. Ajuste a sensibilidade, se necessário.
6. Escolha a aceleração de hardware disponível.
7. Clique em **Start**.
8. Aguarde o processamento terminar.

Os resultados serão criados dentro da pasta escolhida, em uma subpasta com data, modo, sensibilidade e aceleração usada.

## Sensibilidade

- **Low**: detecta menos cortes. Melhor para vídeos calmos ou quando você quer evitar cortes falsos.
- **Normal**: equilíbrio entre precisão e quantidade de cortes.
- **High**: detecta mais cortes. Melhor para vídeos rápidos, trailers, clipes e conteúdos com muita ação.
- **Auto**: tenta escolher parâmetros automaticamente com base no vídeo.

## Aceleração

O app pode usar diferentes opções dependendo do computador e do modo:

- **CPU**: opção mais compatível.
- **NVIDIA**: pode acelerar algumas tarefas em placas NVIDIA.
- **AMD / Intel / Apple**: podem ser usadas em alguns modos de codificação, dependendo do FFmpeg e do sistema.

Nem toda aceleração funciona em todos os modos. Quando uma opção não estiver disponível, o app usa CPU como alternativa.

## Formatos suportados

O app aceita vídeos como:

- `.mp4`
- `.mkv`
- `.mov`
- `.avi`
- `.webm`
- `.m4v`

Arquivos inválidos, temporários ou corrompidos podem ser ignorados ou reparados automaticamente quando possível.

## Requisitos

Para rodar pelo código fonte:

- Python 3.11
- FFmpeg
- FFprobe
- Dependências do `requirements.txt`

Instalação para desenvolvimento:

```bash
pip install -r requirements.txt
```

Execução:

```bash
python Scenespy.py
```

No Windows, também é possível abrir:

```bash
python Scenespy.pyw
```

## FFmpeg e FFprobe

Scenespy precisa do FFmpeg e do FFprobe para ler, validar e cortar vídeos.

O app procura primeiro em:

```text
bin/
```

Depois procura no `PATH` do sistema.

## Prós

- Interface simples para tarefas que normalmente exigem terminal.
- Suporte a múltiplos vídeos em fila.
- Modos diferentes para necessidades diferentes.
- Pode usar aceleração de hardware quando disponível.
- Mantém metadata dos resultados gerados.
- Usa fonte e assets embutidos para uma aparência mais consistente.

## Limitações

- Vídeos muito grandes podem demorar bastante.
- O modo de rostos é mais pesado e depende de bibliotecas de IA.
- Alguns vídeos corrompidos podem falhar mesmo após tentativa de reparo.
- Aceleração de hardware depende do sistema, drivers e suporte do FFmpeg.
- A detecção automática de cenas não é perfeita e pode variar conforme o tipo de vídeo.

## Estrutura do projeto

```text
scenespy/
  assets/
    fonts/
    images/
  app.py
  face_engine.py
  scene_analysis.py
  scene_engine.py
  shared.py
  widgets.py

models/
  yolov8n-face.pt

bin/
  README.md

Scenespy.py
Scenespy.pyw
requirements.txt
```

## Distribuição

Para usuários leigos, o ideal é distribuir uma versão pronta para Windows, preferencialmente como uma pasta com `.exe` ou um instalador.

Uma versão empacotada evita que o usuário precise instalar Python, dependências, FFmpeg ou configurar ambiente virtual.

## Licença

Veja o arquivo [LICENSE](LICENSE).
