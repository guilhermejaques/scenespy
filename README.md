# Scenespy

Scenespy é um app desktop para separar vídeos em partes automaticamente. Ele pode detectar mudanças de cena, cortar por intervalo fixo de tempo ou extrair rostos encontrados no vídeo.

O objetivo é simples: escolher um vídeo, escolher uma pasta de saída, selecionar o modo de corte e deixar o app processar.

## Imagens do app

Face detection and cropping from video:

<img src="https://private-user-images.githubusercontent.com/159738624/594887552-222c8977-bdea-41b0-8aa6-9ab088195f5f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDk4NjAsIm5iZiI6MTc3OTIwOTU2MCwicGF0aCI6Ii8xNTk3Mzg2MjQvNTk0ODg3NTUyLTIyMmM4OTc3LWJkZWEtNDFiMC04YWE2LTlhYjA4ODE5NWY1Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNjUyNDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00MzVhMjUzOTJjZGIwODE1NzU2YjdmOWEyYTU3NmFmODViMDNiMjZhNmEzMWU3OTBhOTFhZmFhY2UzYzhjM2U3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.rS6BvZj52rZkBjriGT3GLhhAdlYYoNF5sqD2Fh0JLyo" width="65%">

<img src="https://private-user-images.githubusercontent.com/159738624/594887551-8bdfd2b3-a0b2-4244-97c0-bc078a1ff509.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDk4NjAsIm5iZiI6MTc3OTIwOTU2MCwicGF0aCI6Ii8xNTk3Mzg2MjQvNTk0ODg3NTUxLThiZGZkMmIzLWEwYjItNDI0NC05N2MwLWJjMDc4YTFmZjUwOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNjUyNDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yZjliZjI5MDkxZmEyMGExMGMwODdkNDIyOGM0ZjBlYzFhMjFhOWVjZjA1NTQyMzBhZWIzZDlkM2UxYTk4NjY3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.r6DXsyZS5kGwZ7gSMTnOynajA0NNlzv0BOC7nfOgZjo" width="40%">

***
Detection and cutting of video scenes:

<img src="https://private-user-images.githubusercontent.com/159738624/594885670-649c86c9-1149-4148-a3ef-d1ecda397edd.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDk4NjUsIm5iZiI6MTc3OTIwOTU2NSwicGF0aCI6Ii8xNTk3Mzg2MjQvNTk0ODg1NjcwLTY0OWM4NmM5LTExNDktNDE0OC1hM2VmLWQxZWNkYTM5N2VkZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNjUyNDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yZTgwMzVjMzI0NTIzYjBiZGU3ZmY5ZDRhODU3N2Y2ZGJiNDA5MDE4MmZlMzBiOTFhYmUxMGY5YTBjNDAxOWM4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.jYCGTleLn67XW6OJpb4qMwg8j-WWIYCE6cYVOXIYUOY" width="65%">

<img src="https://private-user-images.githubusercontent.com/159738624/594885671-ae2498f9-720c-4b2e-b678-757703e9219d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDk4NjUsIm5iZiI6MTc3OTIwOTU2NSwicGF0aCI6Ii8xNTk3Mzg2MjQvNTk0ODg1NjcxLWFlMjQ5OGY5LTcyMGMtNGIyZS1iNjc4LTc1NzcwM2U5MjE5ZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNjUyNDVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yNDE4N2NkZjhhY2RmYTY5NzgzNTg1ZGJhYWJjNTgxMDA0ZmJmZDZjODU0YTg5NzI0YjM1ZDRmZGJkYTBlZTU1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.G4PacTa4uYfHEHqY0gXyT1b2mnXUGrrFwQtdPg4MB4E" width="40%" alt="Output">
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

Procura rostos no vídeo e salva imagens dos rostos detectados. Esse modo usa PyTorch, Ultralytics YOLO e MediaPipe, por isso é mais pesado que os modos de corte de vídeo.

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
- **Auto**: tenta escolher parâmetros automaticamente com base no vídeo. Não é usado no modo de rostos.

## Aceleração

O app separa dois tipos de aceleração:

- **Codificação de vídeo**: usada nos modos **Scene detection** e **Every seconds** quando o app precisa reencodar cortes precisos.
- **Inferência de IA**: usada no modo **Detect faces** para rodar o modelo de detecção de rostos.

Opções disponíveis:

- **CPU**: opção mais compatível. Funciona em todos os modos, mas pode ser mais lenta.
- **NVIDIA**: pode acelerar a codificação via FFmpeg/NVENC e também pode acelerar o modo de rostos via CUDA, se PyTorch com CUDA estiver instalado.
- **AMD**: pode acelerar codificação de vídeo via FFmpeg/AMF em sistemas compatíveis. Não acelera o modo de rostos neste app.
- **Intel**: pode acelerar codificação de vídeo via FFmpeg/QSV em sistemas compatíveis. Não acelera o modo de rostos neste app.
- **Apple**: pode acelerar codificação de vídeo via FFmpeg/VideoToolbox no macOS. Não acelera o modo de rostos neste app.

Nem toda aceleração funciona em todos os computadores. Quando uma opção não está disponível, o app volta para CPU quando possível.

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

- Python 3.11.
- FFmpeg e FFprobe disponíveis no `PATH` ou em `bin/<sistema>/`.
- Dependências do `requirements.txt`.
- O arquivo do modelo `models/yolov8n-face.pt` para o modo **Detect faces**.

Dependências Python principais:

- `customtkinter`: interface desktop.
- `pillow`: imagens e prévias.
- `numpy`: processamento numérico.
- `opencv-contrib-python`: leitura de frames, análise visual e dependência do MediaPipe.
- `av`: backend PyAV usado pelo PySceneDetect.
- `scenedetect`: detecção base de mudanças de cena.
- `torch` e `torchvision`: necessários para o modo **Detect faces**.
- `ultralytics`: carregamento do modelo YOLO de faces.
- `mediapipe`: validação e landmarks faciais.

## Instalação rápida

Crie e ative um ambiente virtual antes de instalar as dependências.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Execução:

```bash
python Scenespy.py
```

No Windows, também é possível abrir sem manter o console visível:

```bash
python Scenespy.pyw
```

## Instalação com CPU

Use esta opção se você não precisa de CUDA ou quer a instalação mais simples.

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

## Instalação com NVIDIA CUDA

CUDA só afeta o modo **Detect faces** quando PyTorch foi instalado com suporte à sua versão de CUDA. Ela também pode ajudar nos cortes de vídeo se o FFmpeg tiver suporte a NVENC.

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

## NVIDIA sem CUDA

Mesmo sem PyTorch CUDA, a opção **NVIDIA** ainda pode acelerar cortes de vídeo se o FFmpeg tiver encoder NVENC (`h264_nvenc`) disponível.

Isso significa:

- **Scene detection** e **Every seconds** podem usar NVIDIA para codificação.
- **Detect faces** volta para CPU se `torch.cuda.is_available()` for `False`.

## AMD

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

## Intel

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

## Apple Silicon e macOS

No macOS, o app pode usar codificação via VideoToolbox (`h264_videotoolbox`) quando o FFmpeg instalado oferece esse encoder.

Instalação recomendada:

```bash
brew install python@3.11 ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Teste do encoder:

```bash
ffmpeg -hide_banner -encoders | grep h264_videotoolbox
```

Observação: o modo **Detect faces** usa PyTorch em CPU no macOS nesta versão do app. A opção **Apple** é para codificação de vídeo, não para inferência facial.

## Linux

Instale Python, ambiente virtual, Tkinter e FFmpeg pelo gerenciador da sua distribuição.

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-tk ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Fedora:

```bash
sudo dnf install python3 python3-tkinter ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Arch Linux:

```bash
sudo pacman -S python tk ffmpeg
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Em algumas distribuições, o pacote `ffmpeg` dos repositórios oficiais pode não incluir todos os encoders de hardware. Se AMD, Intel ou NVIDIA não aparecerem no app, verifique os encoders disponíveis com:

```bash
ffmpeg -hide_banner -encoders
```

## Windows

Instale Python 3.11 e marque a opção para adicionar Python ao `PATH`.

Depois:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Scenespy.py
```

Se o PowerShell bloquear a ativação do ambiente virtual:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Se você estiver usando uma versão pronta do app, ela pode incluir FFmpeg e FFprobe em:

```text
bin/windows/ffmpeg.exe
bin/windows/ffprobe.exe
```

## FFmpeg e FFprobe

Scenespy precisa do FFmpeg e do FFprobe para ler, validar e cortar vídeos.

No Windows, a versão pronta do app pode incluir esses arquivos automaticamente:

```text
bin/windows/ffmpeg.exe
bin/windows/ffprobe.exe
```

No repositório, esses binários não precisam ficar versionados. Para montar uma release, mantenha uma cópia local em `release-assets/windows/ffmpeg/`, copie para `bin/windows/` dentro da pasta final do app e gere o `.zip`.

Os binários Windows recomendados vêm do FFmpeg essentials build da Gyan.dev. Os arquivos de licença do FFmpeg devem acompanhar qualquer distribuição do app.

No Linux e no macOS, a recomendação é instalar pelo sistema:

```bash
sudo apt install ffmpeg
```

```bash
brew install ffmpeg
```

Ao iniciar, o Scenespy procura primeiro em `bin/<sistema>/`. Se não encontrar, procura no `PATH` do sistema.

## Verificação da instalação

Depois de instalar, rode:

```bash
python -m pip check
python -c "import customtkinter, PIL, numpy, cv2, av, scenedetect, ultralytics, mediapipe, torch; print('ok')"
ffmpeg -version
ffprobe -version
```

Se todos os comandos funcionarem, a instalação básica está pronta.

Para verificar CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Para verificar encoders de hardware do FFmpeg:

```bash
ffmpeg -hide_banner -encoders
```

Procure por:

- `h264_nvenc` para NVIDIA.
- `h264_amf` para AMD.
- `h264_qsv` para Intel.
- `h264_videotoolbox` para Apple/macOS.

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
- AMD, Intel e Apple aceleram codificação de vídeo, mas não aceleram a inferência facial nesta versão.

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

Ao distribuir uma versão com FFmpeg incluído, mantenha também os arquivos de licença e informações da build usada.

## Licença

Veja o arquivo [LICENSE](LICENSE).
