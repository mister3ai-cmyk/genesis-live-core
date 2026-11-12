# Business Source License 1.1 - см. research/LICENSE
"""Извлечь из сессии EuRoC MAV только потоки телеметрии (без камер):
mav0/imu0/data.csv, mav0/state_groundtruth_estimate0/data.csv,
mav0/vicon0/data.csv (у сессий Machine Hall vicon0 нет - пропускается).

Источник - одно из:
  --zip  путь к локальному zip сессии (V1_02_medium.zip) или комнаты
         (архив ETH «Vicon Room 1 Datasets» - сессия выбирается по --seq);
  --dir  путь к распакованной папке (где лежит mav0/);
  --url  прямая ссылка на zip: читаются только центральный каталог и
         нужные записи через HTTP Range (единицы МБ вместо гигабайта);
         без поддержки Range сервером - ошибка, а не скачивание целиком.

Результат: research/rt_core/data/euroc/<SEQ>/<поток>.csv (папка в
.gitignore) и SHA-256 каждого файла - для воспроизводимости.

  python3 euroc_fetch.py --seq V1_02_medium --zip ~/Downloads/V1_02_medium.zip
"""
import argparse
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
STREAMS = ("imu0", "state_groundtruth_estimate0", "vicon0")


class HttpRangeFile(io.RawIOBase):
    """Файл только для чтения поверх HTTP Range - чтобы zipfile читал лишь
    каталог и нужные записи."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise OSError("сервер не поддерживает HTTP Range: скачайте zip и используйте --zip")
            self.size = int(r.headers["Content-Length"])
        self.cache = {}
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            lo, hi = i * self.block, min(self.size, (i + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (lo, hi)})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.cache[i] = r.read()
            self.fetched += len(self.cache[i])
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        got = 0
        while got < n:
            i, off = divmod(self.pos, self.block)
            chunk = self._blk(i)[off:off + n - got]
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(fobj, out_dir, seq):
    """Потоки сессии seq из zip. Архив комнаты ETH (2024: «Vicon Room 1
    Datasets» и т. п.) содержит несколько сессий - папками или вложенными
    zip; вложенный zip открывается без распаковки всего архива."""
    done = {}
    with zipfile.ZipFile(fobj) as z:
        names = [n.replace("\\", "/") for n in z.namelist()]
        multi = sum(1 for n in names if n.endswith("mav0/imu0/data.csv")) > 1
        for s in STREAMS:
            cand = [n for n in names if n.endswith("mav0/%s/data.csv" % s) and (not multi or seq in n)]
            if not cand:
                continue
            dst = os.path.join(out_dir, s + ".csv")
            with z.open(cand[0]) as src, open(dst, "wb") as dfh:
                shutil.copyfileobj(src, dfh)
            done[s] = dst
        if not done:
            inner = [n for n in names if n.endswith(seq + ".zip")]
            if inner:
                print("сессия во вложенном архиве: %s" % inner[0])
                with z.open(inner[0]) as zi:
                    return extract_zip(zi, out_dir, seq)
            if not inner:
                print("в архиве нет %s; найдено сессий: %s" % (seq, sorted(
                    {n.split("/")[-4] for n in names if n.endswith("mav0/imu0/data.csv") and n.count("/") >= 3}
                    | {os.path.basename(n)[:-4] for n in names if n.endswith(".zip")})))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="имя сессии, например V1_02_medium")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip")
    g.add_argument("--dir")
    g.add_argument("--url")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "euroc"))
    a = ap.parse_args()
    out_dir = os.path.join(a.out, a.seq)
    os.makedirs(out_dir, exist_ok=True)
    if a.dir:
        done = {}
        for s in STREAMS:
            for base in (a.dir, os.path.join(a.dir, "mav0")):
                p = os.path.join(base, s, "data.csv")
                if os.path.exists(p):
                    done[s] = shutil.copyfile(p, os.path.join(out_dir, s + ".csv"))
                    break
    elif a.zip:
        done = extract_zip(a.zip, out_dir, a.seq)
    else:
        f = HttpRangeFile(a.url)
        done = extract_zip(f, out_dir, a.seq)
        print("скачано %.1f МБ из %.1f МБ архива" % (f.fetched / 1e6, f.size / 1e6))
    if "imu0" not in done or "state_groundtruth_estimate0" not in done:
        sys.exit("не найдены imu0 и state_groundtruth_estimate0 в источнике")
    for s, p in sorted(done.items()):
        print("%-28s %8.1f КБ  sha256 %s" % (s, os.path.getsize(p) / 1e3, _sha(p)))


if __name__ == "__main__":
    main()
