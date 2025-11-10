# Evaluation on LIBERO

We evaluate **X-VLA** on the LIBERO benchmark, which consists of four subtasks: **Spatial**, **Object**, **Goal**, and **Long**.

---

## 1️⃣ Environment Setup

Set up LIBERO following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):
```bash
conda activate X-VLA
python -m deploy \
  --model_path 2toINF/X-VLA-Libero \
  --host 0.0.0.0 \
  --port 8000
```

---


## 3️⃣ Run the Client Evaluation

Launch the LIBERO evaluation client to connect to your X-VLA server:
```bash
cd evaluation/libero
conda activate libero
python eval/libero/libero_client.py \
    --task_suites libero_spatial libero_goal libero_object libero_10 \
    --host 0.0.0.0 \
    --port 12321 \
```


---

## 📊 Results

## 📊 Results

|       |Spatial|Object|Goal|Long|Avg|
| :--------------------: | :--: | :--: | :--: | :--: | :--: | 
|     **Success (%)**    | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
