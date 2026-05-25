# Scenespy

Scenespy é um app desktop para detectar diferentes cenas de um vídeo e separar automaticamente. Ele pode detectar mudanças de cena, cortar por intervalo fixo de tempo ou extrair rostos encontrados no vídeo. 
Ajuda no processo de criação de conteúdo para quem trabalha com vídeo, exige pouco de seu computador e não roda modelos de AI pesados.

O objetivo é simples: escolher um vídeo, escolher uma pasta de saída, selecionar o modo de corte e deixar o app processar. 

---
Face detection and cropping from video:

<img width="65%" height="805" alt="Image" src="https://github.com/user-attachments/assets/222c8977-bdea-41b0-8aa6-9ab088195f5f" />
<img width="65%" height="930" alt="Image" src="https://github.com/user-attachments/assets/8bdfd2b3-a0b2-4244-97c0-bc078a1ff509" />


---
Detection and cutting of video scenes:

<img width="65%" height="800" alt="Image" src="https://github.com/user-attachments/assets/649c86c9-1149-4148-a3ef-d1ecda397edd" />
<img width="65%" height="560" alt="Image" src="https://github.com/user-attachments/assets/ae2498f9-720c-4b2e-b678-757703e9219d" />

## O que o app faz

- Detecta e corta cenas de video.
- Divide vídeos em segmentos por intervalo de segundos.
- Detecta rostos e salva imagens dos melhores recortes encontrados.
- Processa um ou vários vídeos em fila.
- Cria uma pasta de saída organizada para cada processamento.
- Gera arquivos de metadata como `scenes.json` e, quando necessário, `cut_errors.json`.

## Como usar

1. Abra o Scenespy.
2. Em **Source video**, selecione um ou mais vídeos.
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

## Modos disponíveis

### Scene detection

Analisa o vídeo e tenta encontrar mudanças naturais de cena para cortar. É útil para filmes, séries, trailers, vídeos de gameplay, sem limite de tamanho. 
É o modo que mais exige da máquina por causa do pipeline estatístico que tenta encontrar a diferença entre uma cena e outra.

### Every seconds

Corta o vídeo em partes com duração fixa. É o modo mais previsível: você escolhe o intervalo em segundos e o app divide o vídeo. 

### Detect faces

Procura rostos no vídeo e salva imagens dos rostos detectados. É um modo que pode percorrer cada frame do vídeo para encontrar rostos, até mesmo aqueles difíceis de visualizar.
Não faz classificação por pessoa, salva o rosto de uma mesma pessoa em momentos diferentes.

## Sensibilidade

- **Low**: detecta menos cortes. Melhor para vídeos calmos ou quando você quer evitar cortes falsos e fora de contexto. Pode ser bom para filmes, documentários, vídeos onde as sequências são mais longas.
- **Normal**: equilíbrio entre precisão e quantidade de cortes. Teste em seu vídeo e veja os resultados. 
- **High**: detecta mais cortes. Melhor para vídeos rápidos, trailers, clipes e conteúdos com muita ação.
- **Auto**: tenta escolher parâmetros automaticamente com base no vídeo. Não é usado no modo de rostos.

IMPORTANTE: Cada vídeo é único, portanto se um modo de sensibilidade funcionou para um video, nao quer dizer que funcionará em outro video. O teste sempre é a melhor solução.

## Aceleração
- **CPU**: opção mais compatível (padrão). Funciona em todos os modos, mas pode ser mais lenta.
- **NVIDIA**: pode acelerar a codificação via FFmpeg/NVENC e também pode acelerar o modo de rostos via CUDA, se PyTorch com CUDA estiver instalado. 
- **AMD**: pode acelerar codificação de vídeo via FFmpeg/AMF em sistemas compatíveis. Não acelera o modo de rostos.
- **Intel**: pode acelerar codificação de vídeo via FFmpeg/QSV em sistemas compatíveis. Não acelera o modo de rostos.
- **Apple**: pode acelerar codificação de vídeo via FFmpeg/VideoToolbox no macOS. Não acelera o modo de rostos. 

Hoje a forma mais relevante de acelerar o processamento é com NVIDIA CUDA, mas o app funcionará bem caso você não use CUDA, fique tranquilo.  

### Formatos suportados

O app aceita vídeos como:

- `.mp4` O formato mais compatível.
- `.mkv` 
- `.mov`
- `.avi`
- `.webm`
- `.m4v`

Arquivos inválidos, temporários ou corrompidos podem ser ignorados ou reparados automaticamente quando possível. 
MKV suporta múltiplos áudios e pode apresentar problemas no container, por isso, em vídeos difíceis de processar, o app converterá de MKV para MP4 para tentar resolver o problema.

---
# Instalação

## Instalação rápida | GitHub Releases

Use a versão pronta do Scenespy na aba **Releases** do GitHub. Não use o botão **Code > Download ZIP** se você quer apenas instalar e usar o app. Já existe um pacote de release para cada sistema operacional suportado:

- Windows: [Scenespy-Windows-x64](https://github.com/guilhermejaques/scenespy/releases/tag/0.1.0)
- Linux: [Scenespy-Linux-x64](https://github.com/guilhermejaques/scenespy/releases/tag/0.1.0)
- macOS: [Scenespy-MacOS-x64](https://github.com/guilhermejaques/scenespy/releases/tag/0.1.0)

Baixe o pacote para o seu sistema, extraia a pasta e rode o instalador `install_runtime` que acompanha o app. Esse instalador configura e instala as dependências externas usadas pelo app, como FFmpeg/FFprobe, Python privado e pacotes de IA que são necessários. Abra a linha de comando em seu sistema e localize o diretório do app para rodar o instalador, depois rode o app `Scenespy`. 

Windows (instalador .bat pode rodar como administrador, use o botão direito para isso)

```bat
install_runtime_windows.bat 
Scenespy.exe
```

Linux e Mac OS:

```bash
chmod +x install_runtime.sh  # Comando para permissão
./install_runtime.sh 
./Scenespy # App 
```

Para usuários iniciantes que não sabem rodar a linha de comando:
você só precisa localizar a pasta onde está o app e rodar o instalador antes de executar o app. Veja um exemplo:

`cd Downloads` >  `cd Scenespy-Linux-x64` > `chmod +x install_runtime.sh` > `./install_runtime.sh`

### Comandos para usar no terminal:
Abrir terminal

| Sistema | Como abrir |
|---|---|
| Windows | `Win + R` → `cmd` |
| PowerShell | Pesquisar “PowerShell” |
| macOS | `Command + Espaço` → `Terminal` |
| Linux | `Ctrl + Alt + T` |

---

Ver em qual pasta você está

| Sistema | Comando |
|---|---|
| Windows CMD | `cd` |
| PowerShell | `pwd` |
| macOS/Linux | `pwd` |

---

Listar arquivos

| Sistema | Comando |
|---|---|
| Windows CMD | `dir` |
| PowerShell | `ls` |
| macOS/Linux | `ls` |

---

Entrar em uma pasta

| Sistema | Comando |
|---|---|
| Windows | `cd Downloads` |
| macOS/Linux | `cd Downloads` |

---

Voltar uma pasta

| Sistema | Comando |
|---|---|
| Todos | `cd ..` |

---

Rodar um arquivo
| Sistema | Comando |
|---|---|
| Todos | `./ARQUIVO.SH` |


## Rodar pelo código-fonte

Você é responsável por instalar Python, dependências Python, FFmpeg/FFprobe e bibliotecas.

Requisitos para código-fonte:

- Python 3.11.
- FFmpeg e FFprobe.
- Dependências do `requirements.txt`.
- No Windows, Microsoft Visual C++ Redistributable x64 pode ser necessário para o PyTorch.

No Arch, use `pyenv` ou outro método equivalente para garantir Python 3.11, porque a versão `python` dos repositórios pode ser mais nova que a suportada pelas dependências de IA.

Dependências Python principais instaladas por `requirements.txt`:

- `customtkinter`: interface desktop.
- `pillow`: imagens e prévias.
- `numpy`: processamento numérico.
- `opencv-contrib-python`: leitura de frames, análise visual e dependência do MediaPipe.
- `av`: backend PyAV usado pelo PySceneDetect.
- `scenedetect`: detecção base de mudanças de cena.
- `torch` e `torchvision`: necessários para o modo **Detect faces**.
- `ultralytics`: carregamento do modelo YOLO de faces.
- `mediapipe`: validação e landmarks faciais.

### Instalação para CPU apenas 

Use esta opção se você não precisa de CUDA NVIDIA ou quer a instalação mais simples.

```bash
python -m pip install -r requirements.txt
```

Se você quiser garantir uma instalação estritamente CPU para PyTorch:

```bash
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Com CPU:

- **Scene detection** funciona.
- **Every seconds** funciona.
- **Detect faces** funciona, mas pode ser mais lento.
- Aceleração NVIDIA/AMD/Intel/Apple pode não aparecer ou pode não ser usada.

### Instalação com NVIDIA CUDA 

CUDA só afeta o modo **Detect faces** quando PyTorch foi instalado com suporte à sua versão de CUDA. Ela também pode ajudar nos cortes de vídeo.

O `requirements.txt` fixa `torch==2.5.1` e `torchvision==0.20.1` sem escolher uma build CUDA específica. Para CUDA, instale PyTorch a partir do índice oficial da versão desejada.

Exemplo para CUDA 12.1:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Depois, confirme se CUDA foi detectada:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

O resultado esperado para CUDA ativa é:

```text
True
```

Se retornar `False`, o app ainda funciona em CPU, mas o modo **Detect faces** não usará a GPU.

### NVIDIA sem CUDA

Mesmo sem PyTorch CUDA, a opção **NVIDIA** ainda pode acelerar cortes de vídeo se o FFmpeg tiver encoder NVENC (`h264_nvenc`) disponível.

Isso significa:

- **Scene detection** e **Every seconds** podem usar NVIDIA para codificação.
- **Detect faces** volta para CPU se `torch.cuda.is_available()` for `False`.

### AMD

No Scenespy, AMD é usada para codificação de vídeo via FFmpeg/AMF (`h264_amf`) quando disponível.

Requisitos práticos:

- GPU AMD compatível.
- Driver AMD instalado.
- FFmpeg compilado com suporte a AMF.

AMD não acelera o modo **Detect faces** neste app. O modo de rostos usa CPU ou NVIDIA CUDA.

Para testar se o FFmpeg reconhece AMF:

```bash
ffmpeg -hide_banner -encoders | grep h264_amf
```

No Windows PowerShell:

```powershell
ffmpeg -hide_banner -encoders | Select-String h264_amf
```

### Intel

No Scenespy, Intel é usada para codificação de vídeo via FFmpeg/QSV (`h264_qsv`) quando disponível.

Requisitos práticos:

- CPU/GPU Intel com Quick Sync Video.
- Driver Intel atualizado.
- FFmpeg compilado com suporte a QSV.

Intel não acelera o modo **Detect faces** neste app.

Teste:

```bash
ffmpeg -hide_banner -encoders | grep h264_qsv
```

No Windows PowerShell:

```powershell
ffmpeg -hide_banner -encoders | Select-String h264_qsv
```

### Apple Silicon e macOS

No macOS, o app pode usar codificação via VideoToolbox (`h264_videotoolbox`) quando o FFmpeg instalado oferece esse encoder.

Teste do encoder:

```bash
ffmpeg -hide_banner -encoders | grep h264_videotoolbox
```

Observação: o modo **Detect faces** usa PyTorch em CPU no macOS nesta versão do app. A opção **Apple** é para codificação de vídeo, não para inferência facial.

### FFmpeg e FFprobe

Scenespy precisa do FFmpeg e do FFprobe para ler, validar e cortar vídeos.

Nas versões prontas da aba **Releases**, o instalador de runtime baixa ou instala FFmpeg/FFprobe automaticamente.

No Windows, `install_runtime_windows.bat` instala os binários do FFmpeg essentials build da Gyan.dev em `%LOCALAPPDATA%/Scenespy/runtime/`.

No Linux e no macOS, a recomendação é instalar pelo sistema:

```bash
sudo apt install ffmpeg
```

```bash
brew install ffmpeg
```

Ao iniciar, o Scenespy procura primeiro em `bin/<sistema>/`. Se não encontrar, procura no `PATH` do sistema.

## Verificação da instalação pelo código-fonte

Depois de instalar pelo código-fonte, rode:

```bash
python -m pip check
python -c "import customtkinter, PIL, numpy, cv2, av, scenedetect, ultralytics, mediapipe, torch; print('ok')"
ffmpeg -version
ffprobe -version
```

Se todos os comandos funcionarem, a instalação básica está pronta.

## Problemas comuns

### `ffmpeg` ou `ffprobe` não encontrado

Instale FFmpeg e garanta que os executáveis estejam no `PATH`, ou coloque os binários em `bin/<sistema>/`.

### Modo Detect faces não abre

Verifique se `torch`, `ultralytics`, `mediapipe` e o modelo `models/yolov8n-face.pt` existem.

```bash
python -c "import torch, ultralytics, mediapipe; print('ok')"
```

### CUDA não aparece

Verifique se você instalou uma build CUDA do PyTorch e se o driver NVIDIA está atualizado.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### AMD, Intel ou Apple não aparecem

Essas opções dependem do FFmpeg, dos drivers e do sistema operacional. Confira se o encoder correspondente existe:

```bash
ffmpeg -hide_banner -encoders
```

### Instalação falha no MediaPipe

Use Python 3.11. Algumas versões de Python podem não ter wheels compatíveis para todas as dependências.

### Existem `opencv-python` e `opencv-contrib-python` instalados

O app declara `opencv-contrib-python` porque o MediaPipe depende dele. O Ultralytics também pode instalar `opencv-python`. Se `python -m pip check` não apontar conflito e o app abrir normalmente, não há ação obrigatória.

